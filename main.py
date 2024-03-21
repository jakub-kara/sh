import os, sys
import pickle, json
import numpy as np

from abinitio import update_potential_energy
from classes import Trajectory
from selection import select_est, select_force_updater, select_nac_setter, select_coeff_propagator, select_solvers, select_stepfunc
from io_methods import finalise_dynamics, write_headers, write_dat, write_xyz, write_log, write_mat, time_log, step_log, back_up_step, Printer
from integrators import shift_values, est_wrapper, calculate_am4_coeffs, calculate_sy4_coeffs, update_tdc, integrate_quantum, get_dt, RKN8
from hopping import adjust_velocity_and_hop, decoherence_edc
from constants import Constants


"SH with Robust Error and Coupling Control"
def main():
    #print("I'm currently broken, please wait until Jakub fixes me.")
    #exit(1)

    inp = sys.argv[1]   
    if inp.endswith(".pkl"):
        with open(inp, "rb") as traj_pkl:
            traj = pickle.load(traj_pkl)
            write_log(traj, "Restarting from backup at the start of interrupted step.\n")
    else:
        with open(inp, "r") as infile: config = json.load(infile)
        traj = Trajectory(config) 

        initialise_dynamics(traj)

    loop_dynamics(traj)
    finalise_dynamics(traj)

def get_inp(x: np.ndarray):
    inp = 0
    n = x.shape[0]
    for i in range(n):
        for j in range(i):
            inp += np.sum(x[i,j]**2)
    return inp

def initialise_dynamics(traj: Trajectory):
    select_est(traj)
    select_force_updater(traj)
    select_nac_setter(traj)
    select_coeff_propagator(traj)
    select_stepfunc(traj)
    select_solvers(traj)

    write_headers(traj)
    traj.est.nacs_setter(traj, False)
    time_log(traj, "Initial EST: ", lambda : traj.est.run(traj))
    traj.est.first = False
    traj.geo.force_updater(traj)
    update_tdc(traj)
    traj.pes.nac_ddt_mnss[:-1] = traj.pes.nac_ddt_mnss[-1]
    update_potential_energy(traj)

    traj.ctrl.dt = traj.ctrl.dt_func(traj, get_inp(traj.pes.nac_ddt_mnss[-1,0]))
    traj.ctrl.dtq = traj.ctrl.dt/traj.par.n_qsteps
    traj.ctrl.h[-1] = traj.ctrl.dt

    back_up_step(traj)
    write_dat(traj)
    write_xyz(traj)

def check_est(traj: Trajectory):
    if traj.ctrl.curr_step < 2: return True, 0
    ke0 = 0.5*np.sum(traj.geo.mass_a[:,None]*traj.geo.velocity_mnad[-2,0]**2)
    ke1 = 0.5*np.sum(traj.geo.mass_a[:,None]*traj.geo.velocity_mnad[-1,0]**2)
    pe0 = traj.pes.poten_mn[-2,0]
    pe1 = traj.pes.poten_mn[-1,0]
    diff = np.abs(ke0 + pe0 - ke1 - pe1)

    if diff < traj.ctrl.en_thresh[max(traj.ctrl.conv_status-1, 0)]: return True
    else:
        conv_status = traj.ctrl.conv_status + 4 - len(np.trim_zeros(traj.ctrl.en_thresh))
        if traj.ctrl.conv_status >= 3:
            print("Maximum energy difference exceeded on RKN8.")
            print("Terminating trajectory.")

            write_log(traj, "Maximum energy difference exceeded on RKN8\n")
            write_log(traj, "Terminating trajectory\n")
            exit(21)

        os.system(f"cp backup/{traj.est.program}.wf est/")
        print(f"Energy not conserved. Energy difference: {diff*Constants.eh2ev} eV.")
        write_log(traj, f"Energy not conserved. Energy difference: {diff*Constants.eh2ev} eV.\n")
        write_log(traj, f"Convergence status {conv_status}. Stepping back.\n")

        with open("backup/traj.pkl", "rb") as traj_pkl:
            traj = pickle.load(traj_pkl)
            traj.ctrl.conv_status = conv_status
        
        return False
                

def solver_wrapper(traj: Trajectory, solver, scheme):
    traj.geo.position_mnad[-1,0], traj.geo.velocity_mnad[-1,0], traj.geo.force_mnad[-1,0] = \
        solver(
            traj.geo.position_mnad[-scheme.m-1:-1,0], 
            traj.geo.velocity_mnad[-scheme.m-1:-1,0], 
            traj.geo.force_mnad[-scheme.m-1:-1,0], 
            est_wrapper, (traj,), traj.ctrl.dt, scheme)

def loop_dynamics(traj: Trajectory):
    while traj.ctrl.curr_time < traj.ctrl.t_max:
        print(traj.ctrl.curr_time)
        if os.path.isfile("stop"):
            exit(23)
        
        step_log(traj)
        shift_values(traj.geo.position_mnad, traj.geo.velocity_mnad, traj.geo.force_mnad)
        shift_values(traj.pes.ham_diag_mnss, traj.pes.nac_ddr_mnssad, traj.pes.nac_ddt_mnss)
        shift_values(traj.est.coeff_mns, traj.pes.poten_mn)
        
        if traj.ctrl.init_steps > 0 or traj.ctrl.conv_status == 1:
            time_log(traj, "Classical + EST: ", lambda : solver_wrapper(traj, traj.geo.init_solver, traj.geo.init_scheme))
        elif traj.ctrl.conv_status == 2:
            time_log(traj, "Classical + EST: ", lambda : solver_wrapper(traj, traj.geo.init_solver, RKN8))
        else:
            if "sy" in traj.geo.scheme_name: traj.geo.loop_scheme = calculate_sy4_coeffs(traj.ctrl.h[-4], traj.ctrl.h[-3], traj.ctrl.h[-2], traj.ctrl.h[-1])
            time_log(traj, "Classical + EST: ", lambda : solver_wrapper(traj, traj.geo.loop_solver, traj.geo.loop_scheme))
        if traj.ctrl.init_steps > 0: traj.ctrl.init_steps -= 1

        traj.est.calculate_nacs[:] = False

        update_potential_energy(traj)

        if not check_est(traj): continue
        traj.ctrl.conv_status = 0

        # step 4&5: update electronic wf coefficients & compute hopping probabilities
        time_log(traj, "WF coeffs: ", lambda : update_tdc(traj), lambda: integrate_quantum(traj))

        # step 6: select a new active state
        if traj.par.type == "sh":
            adjust_velocity_and_hop(traj)
            if traj.hop.decoherence == "edc":
                decoherence_edc(traj)

        shift_values(traj.ctrl.h)
        #traj.ctrl.dt = 1/(2/traj.ctrl.dt_func(traj, get_inp(traj.pes.nac_ddt_mnss[-1,0])) - 1/traj.ctrl.h[-2])
        #traj.ctrl.dt = 2*traj.ctrl.dt_func(traj, get_inp(traj.pes.nac_ddt_mnss[-1,0])) - traj.ctrl.h[-2]
        get_dt(traj)
        traj.ctrl.dtq = traj.ctrl.dt/traj.par.n_qsteps
        #traj.ctrl.h[-1] = traj.ctrl.dt
        print(traj.ctrl.h)

        
        # step 7: recalculate EST if hop occured
        if np.any(traj.est.calculate_nacs):
            traj.est.run(traj)
            traj.geo.force_updater(traj)
            traj.est.calculate_nacs[:] = False
            update_potential_energy(traj)

        
        write_dat(traj)
        write_xyz(traj)
        time_log(traj, "Saving matrices: ", lambda : write_mat(traj))

        traj.ctrl.curr_time += traj.ctrl.dt
        traj.ctrl.curr_step += 1
        time_log(traj, "Backup: ", lambda : back_up_step(traj))


if __name__ == "__main__":
    main()
