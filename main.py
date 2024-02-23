import os, sys
import pickle, json
import numpy as np
from copy import deepcopy

from classes import Trajectory
from selection import select_est, select_force_updater, select_nac_setter, select_coeff_propagator, select_solvers, select_stepfunc
from io_methods import finalise_dynamics, write_headers, write_dat, write_xyz, write_log, time_log, step_log, back_up_step, Printer
from integrators import shift_values, est_wrapper, calculate_am4_coeffs, calculate_sy4_coeffs, update_tdc, integrate_quantum, get_dt
from hopping import adjust_velocity_and_hop, decoherence_edc
from constants import Constants


"SH with Robust Error and Coupling Control"
def main():
    # IMPLEMENT RESTART
    inp = sys.argv[1]   
    pkl = False
    if inp.endswith(".pkl"):
        with open("backup/traj.pkl", "rb") as traj_pkl:
            traj = pickle.load(traj_pkl)
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
    traj.est.nacs_setter(traj, True)
    time_log(traj, "Initial EST: ", lambda : traj.est.run(traj))
    traj.est.first = False
    traj.geo.force_updater(traj)
    update_tdc(traj)
    traj.pes.nac_ddt_mnss[:-1] = traj.pes.nac_ddt_mnss[-1]

    back_up_step(traj)
    write_dat(traj)
    write_xyz(traj)
    step_log(traj)

    #figure out a place to store pe
    if traj.par.type == "sh":
        traj.pes.poten_mn[-1,0] = traj.pes.ham_diag_mnss[-1,0,traj.hop.active, traj.hop.active]
    elif traj.par.type == "mfe":
        traj.pes.poten_mn[-1,0] = 0
        for s in range(traj.par.n_states):
            traj.pes.poten_mn[-1,0] += np.abs(traj.est.coeff_mns[-1,0,s])**2 * traj.pes.ham_diag_mnss[-1,0,s,s]
    
    traj.ctrl.dt = traj.ctrl.dt_func(traj, get_inp(traj.pes.nac_ddt_mnss[-1,0]))
    traj.ctrl.dtq = traj.ctrl.dt/traj.par.n_qsteps
    traj.ctrl.h[-1] = traj.ctrl.dt

    with open("data/cons.dat", "w") as f:
        f.write(Printer.write(" Time", "s"))
        f.write(Printer.write("Redo", "s"))
        f.write("\n")

def check_est(traj: Trajectory):
    ke0 = 0.5*np.sum(traj.geo.mass_a[:,None]*traj.geo.velocity_mnad[-2,0]**2)
    ke1 = 0.5*np.sum(traj.geo.mass_a[:,None]*traj.geo.velocity_mnad[-1,0]**2)
    pe0 = traj.pes.poten_mn[-2,0]
    pe1 = traj.pes.poten_mn[-1,0]
    print(ke0, pe0, ke1, pe1)
    return np.abs(ke0 + pe0 - ke1 - pe1) < 2e-4

def loop_dynamics(traj: Trajectory):
    while traj.ctrl.curr_time <= traj.ctrl.t_max:
        traj_old = deepcopy(traj)

        if os.path.isfile("stop"):
            exit(23)

        traj.ctrl.curr_time += traj.ctrl.dt
        traj.ctrl.curr_step += 1
        step_log(traj)

        shift_values(traj.geo.position_mnad, traj.geo.velocity_mnad, traj.geo.force_mnad)
        shift_values(traj.pes.ham_diag_mnss, traj.pes.nac_ddr_mnssad, traj.pes.nac_ddt_mnss)
        shift_values(traj.est.coeff_mns, traj.pes.poten_mn)
        
        if traj.ctrl.init_steps:
            traj.ctrl.init_steps -= 1
            temp = time_log(traj, "Classical + EST: ", 
                lambda : traj.geo.init_solver(traj.geo.position_mnad[-traj.geo.init_scheme.m-1:-1,0],
                                     traj.geo.velocity_mnad[-traj.geo.init_scheme.m-1:-1,0],
                                     traj.geo.force_mnad[-traj.geo.init_scheme.m-1:-1,0],
                                     est_wrapper, (traj,), traj.ctrl.dt,
                                     traj.geo.init_scheme, None))
            tempx, tempv, tempf = temp[0]
            
        else:
            if "sy" in traj.geo.scheme_name:
                traj.geo.loop_scheme_x = calculate_sy4_coeffs(traj.ctrl.h[-4], traj.ctrl.h[-3], traj.ctrl.h[-2], traj.ctrl.h[-1])
                traj.geo.loop_scheme_v = calculate_am4_coeffs(traj.ctrl.h[-3], traj.ctrl.h[-2], traj.ctrl.h[-1])
            temp = time_log(traj, "Classical + EST:  ", 
                lambda : traj.geo.loop_solver(traj.geo.position_mnad[-traj.geo.loop_scheme_x.m-1:-1,0],
                                     traj.geo.velocity_mnad[-traj.geo.loop_scheme_x.m-1:-1,0],
                                     traj.geo.force_mnad[-traj.geo.loop_scheme_x.m-1:-1,0],
                                     est_wrapper, (traj,), traj.ctrl.dt,
                                     traj.geo.loop_scheme_x, traj.geo.loop_scheme_v))
            tempx, tempv, tempf = temp[0]

        traj.est.calculate_nacs[:] = False
        traj.geo.position_mnad[-1,0], traj.geo.velocity_mnad[-1,0], traj.geo.force_mnad[-1,0] = tempx, tempv, tempf
        if traj.par.type == "sh":
            traj.pes.poten_mn[-1,0] = traj.pes.ham_diag_mnss[-1,0,traj.hop.active, traj.hop.active]
        elif traj.par.type == "mfe":
            traj.pes.poten_mn[-1,0] = 0
            for s in range(traj.par.n_states):
                traj.pes.poten_mn[-1,0] = np.abs(traj.est.coeff_mns[-1,0,s])**2 * traj.pes.ham_diag_mnss[-1,0,s,s]
        if not check_est(traj) and traj.ctrl.curr_step > 1:
            with open ("data/cons.dat", "a") as f:
                f.write(Printer.write(traj.ctrl.curr_time*Constants.au2fs, "f"))
                f.write(Printer.write(1, "i"))
                f.write("\n")

            print("Energy not conserved, stepping back")
            traj = traj_old
            traj.ctrl.init_steps = 1
            continue

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
        if traj.par.type == "sh":
            traj.pes.poten_mn[-1,0] = traj.pes.ham_diag_mnss[-1,0,traj.hop.active, traj.hop.active]
        elif traj.par.type == "mfe":
            traj.pes.poten_mn[-1,0] = 0
            for s in range(traj.par.n_states):
                traj.pes.poten_mn[-1,0] = np.abs(traj.est.coeff_mns[-1,0,s])**2 * traj.pes.ham_diag_mnss[-1,0,s,s]

        time_log(traj, "Backup: ", lambda : back_up_step(traj))
        
        write_dat(traj)
        write_xyz(traj)

        back_up_step(traj)

        

if __name__ == "__main__":
    main()