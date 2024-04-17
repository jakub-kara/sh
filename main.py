import os, sys
import pickle, json
import numpy as np

from abinitio import update_potential_energy
from classes import Trajectory
from selection import select_est, select_force_updater, select_nac_setter, select_coeff_propagator, select_solvers, select_stepfunc
from io_methods import finalise_dynamics, write_headers, write_dat, write_xyz, write_log, write_mat, time_log, step_log, back_up_step
from integrators import shift_values, est_wrapper, calculate_sy4_coeffs, update_tdc, integrate_quantum, get_dt, RKN8
from hopping import adjust_velocity_and_hop, decoherence_edc
from constants import Constants


"SH with Robust Error and Coupling Control"
def main():
    """
    Main entry point to the simulation of single-trajectory dynamics.
    Instantiates trajectory object based on the cmdline arguments and starts the dynamics.

    Parameters
    ----------
    None
    
    Returns
    -------
    None
    
    Modifies
    --------
    None
    """
    
    # get first cmdline argument after script name
    inp = sys.argv[1]
    # pickle file contains previously saved trajectory
    if inp.endswith(".pkl"):
        with open(inp, "rb") as traj_pkl:
            # create trajectory object from the pickle data
            traj = pickle.load(traj_pkl)
            write_log(traj, "Restarting from backup at the start of interrupted step.\n")
    # otherwise try to read the supplied file as json
    else:
        with open(inp, "r") as infile: config = json.load(infile)
        # initialise trajectory based on the contents of the input file
        traj = Trajectory(config) 

        # setup required before first step
        initialise_dynamics(traj)

    # perform dynamics iteratively until termination condition met
    loop_dynamics(traj)
    # clean up before program terminates
    finalise_dynamics(traj)

def get_inp(x: np.ndarray):
    inp = 0
    n = x.shape[0]
    for i in range(n):
        for j in range(i):
            inp += np.sum(x[i,j]**2)
    return inp

def initialise_dynamics(traj: Trajectory):
    """
    Setup for the first step of dynamics.
    Carries out initial EST calculation, creates output files, and determines the stepsize.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.est.first

    traj.ctrl.dt

    traj.ctrl.dtq
    """
    

    # TODO: move selectors to class constructors
    select_est(traj)
    select_force_updater(traj)
    select_nac_setter(traj)
    select_coeff_propagator(traj)
    select_stepfunc(traj)
    select_solvers(traj)

    # write headers to appropriate files
    write_headers(traj)
    # request initial EST
    traj.est.nacs_setter(traj, False)
    time_log(traj, "Initial EST: ", lambda : traj.est.run(traj))
    traj.est.first = False
    update_potential_energy(traj)
    # calculate forces
    traj.ctrl.force_updater(traj)
    # calculate time-derivative coupling
    update_tdc(traj)
    # maybe not necessary, for safety
    for m in range(traj.par.n_steps-1):
        for n in range(traj.par.n_substeps):
            traj.pes_mn[m,n].nac_ddt_ss = traj.pes_mn[-1,0].nac_ddt_ss

    # TODO: move to a separate function
    traj.ctrl.dt = traj.ctrl.dt_func(traj, get_inp(traj.pes_mn[-1,0].nac_ddt_ss))
    traj.ctrl.dtq = traj.ctrl.dt/traj.par.n_qsteps
    traj.ctrl.h[-1] = traj.ctrl.dt

    # back up and write outputs
    back_up_step(traj)
    write_dat(traj)
    write_xyz(traj)
    #write_mat(traj)

def check_est(traj: Trajectory) -> Trajectory:
    """
    Calculates the total energy change between previous and current step.
    Restarts the step with a more accurate integrator if energy check not passed.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    traj: Trajectory
    
    Modifies
    --------
    traj
        returns to previous step if criteria not met
    traj.ctrl
    """
    
    # only run if a previous step exists
    if traj.ctrl.curr_step < 2: return traj
    # get kinetic and potential energies
    ke0 = 0.5*np.sum(traj.geo_mn[-2,0].mass_a[:,None]*traj.geo_mn[-2,0].velocity_ad**2)
    ke1 = 0.5*np.sum(traj.geo_mn[-1,0].mass_a[:,None]*traj.geo_mn[-1,0].velocity_ad**2)
    pe0 = traj.pes_mn[-2,0].poten
    pe1 = traj.pes_mn[-1,0].poten
    diff = np.abs(ke0 + pe0 - ke1 - pe1)

    # check energy difference against the threshold correpoding to the current convergence status
    if diff > traj.ctrl.en_thresh[max(traj.ctrl.conv_status-1, 0)]:
        # increment convergence status
        conv_status = traj.ctrl.conv_status + 1
        # check for maximum convergence status
        if traj.ctrl.conv_status >= 3:
            # sorry, cannot do better than that
            print("Maximum energy difference exceeded on RKN8.")
            print("Terminating trajectory.")

            write_log(traj, "Maximum energy difference exceeded on RKN8\n")
            write_log(traj, "Terminating trajectory\n")
            exit(21)

        # copy back the backup WF
        os.system(f"cp backup/{traj.est.program}.wf est/")
        write_log(traj, f"Energy not conserved. Energy difference: {diff*Constants.eh2ev} eV.\n")
        write_log(traj, f"Convergence status {conv_status}. Stepping back.\n")

        # load traj from backup of the previous step
        with open("backup/traj.pkl", "rb") as traj_pkl:
            traj = pickle.load(traj_pkl)
        # set the new convergence status
        traj.ctrl.conv_status = conv_status
    else:
        # if the condition was passed, reset the status
        traj.ctrl.conv_status = 0
    return traj
                

def solver_wrapper(traj: Trajectory, scheme):
    """
    Wraps the nuclear integrator for simpler calls.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    scheme: Any
        integration scheme with all the information outlined in integrators.py

    Returns
    -------
    None
    
    Modifies
    --------
    traj.geo_mn[-1,0].position_ad

    traj.geo_mn[-1,0].velocity_ad

    traj.geo_mn[-1,0].force_ad
    """

    # call the solver in integrators.py or externally defined    
    traj.geo_mn[-1,0].position_ad, traj.geo_mn[-1,0].velocity_ad, traj.geo_mn[-1,0].force_ad = \
        scheme.s(
            np.array([traj.geo_mn[m,0].position_ad for m in range(-scheme.m-1,-1)]),
            np.array([traj.geo_mn[m,0].velocity_ad for m in range(-scheme.m-1,-1)]),
            np.array([traj.geo_mn[m,0].force_ad for m in range(-scheme.m-1,-1)]),
            est_wrapper, (traj,), traj.ctrl.dt, scheme)

def loop_dynamics(traj: Trajectory):
    """
    Controls the iterative cycle in single-trajectory dynamics.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.ctrl.curr_time

    traj.ctrl.curr_step

    
    """

    # check if termination condition met
    while traj.ctrl.curr_time < traj.ctrl.t_max:
        # increment time
        traj.ctrl.curr_time += traj.ctrl.dt
        traj.ctrl.curr_step += 1
        print(traj.ctrl.curr_time)
        print(traj.est.coeff_mns[:,:,1])

        # place a file called "stop" in the working directory to force a termination at the beginning of a new iteration
        if os.path.isfile("stop"):
            exit(23)
        
        step_log(traj)
        # store previous values
        shift_values(traj.pes_mn)
        shift_values(traj.geo_mn)
        
        # decide on the integrator based on the convergence status and initialisation steps
        if traj.ctrl.init_steps > 0 or traj.ctrl.conv_status == 1:
            # typically RKN4
            time_log(traj, "Classical + EST: ", lambda : solver_wrapper(traj, traj.ctrl.nuc_init_scheme))
        elif traj.ctrl.conv_status == 2:
            # hard-coded RKN8
            time_log(traj, "Classical + EST: ", lambda : solver_wrapper(traj, RKN8))
        else:
            # recalculate multistep method coefficients
            if "sy" in traj.ctrl.nuc_scheme_name: traj.ctrl.nuc_loop_scheme = calculate_sy4_coeffs(traj.ctrl.h[-4], traj.ctrl.h[-3], traj.ctrl.h[-2], traj.ctrl.h[-1])
            # standard integrator (typically VV/SY4)
            time_log(traj, "Classical + EST: ", lambda : solver_wrapper(traj, traj.ctrl.nuc_loop_scheme))
        if traj.ctrl.init_steps > 0: traj.ctrl.init_steps -= 1
        
        # move to abinitio
        traj.est.calc_nacs[:] = False
        update_potential_energy(traj)

        # check of energy conserved
        # if not, traj gets restarted from previous step
        traj = check_est(traj)
        # if the check was passed, conv_status would be 0
        if traj.ctrl.conv_status > 0: continue

        # store previous values
        shift_values(traj.est.coeff_mns)
        # integrate WF coefficients and hopping probabilities [in SH]
        time_log(traj, "WF coeffs: ", lambda : update_tdc(traj), lambda: integrate_quantum(traj))

        # check for any hops and make corresponding adjustments
        if traj.par.type == "sh":
            adjust_velocity_and_hop(traj)
            print(traj.est.coeff_mns[:,:,1])

            # perform decoherence correction
            if traj.hop.decoherence == "edc":
                decoherence_edc(traj)
        print(traj.est.coeff_mns[:,:,1])

        # store previous stepsize values
        shift_values(traj.ctrl.h)
        #traj.ctrl.dt = 1/(2/traj.ctrl.dt_func(traj, get_inp(traj.pes.nac_ddt_mnss[-1,0])) - 1/traj.ctrl.h[-2])
        #traj.ctrl.dt = 2*traj.ctrl.dt_func(traj, get_inp(traj.pes.nac_ddt_mnss[-1,0])) - traj.ctrl.h[-2]
        get_dt(traj)
        # update quantum dt
        traj.ctrl.dtq = traj.ctrl.dt/traj.par.n_qsteps
        #traj.ctrl.h[-1] = traj.ctrl.dt
        #print(traj.ctrl.h)

        
        # recalculate EST if hop occured
        if np.any(traj.est.calc_nacs):
            traj.est.run(traj)
            traj.ctrl.force_updater(traj)
            traj.est.calc_nacs[:] = False
            update_potential_energy(traj)
        print(traj.est.coeff_mns[:,:,1])
        

        # write output
        write_dat(traj)
        write_xyz(traj)
        time_log(traj, "Saving matrices: ", lambda : write_mat(traj))

        # back up the step
        time_log(traj, "Backup: ", lambda : back_up_step(traj))

if __name__ == "__main__":
    main()
