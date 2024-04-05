import os
import pickle, h5py
import time

from classes import Trajectory
from kinematics import *
from utility import get_dirs
from constants import Constants

class Printer:
    field_length = 20
    tdict = {
        "f" : (fform := f" < {field_length}.10e"),
        "p" : (pform := f" < {field_length}.4%"),
        "b" : (bform := f" < 6"),
        "s" : (sform := f" <{field_length}"),
        "i" : (iform := f" < {field_length}.0f"),
    }

    @staticmethod
    def write(val, form):
        if form in Printer.tdict.keys():
            return f"{val:{Printer.tdict[form]}}"
        elif form == "z":
            return f"{val.real:{Printer.tdict['f']}}" + f"{val.imag:< {Printer.field_length-1}.10e}" + "j "
        else:
            return f"{val:{form}}"

ensemble =  {
    "generate": True,
    "ntraj": 1,
    "location": ".",
    "idoffset": 0,
    "geom": "geom.xyz",
    "input": "input.inp",
    "wf": "wf.wf",
    "submission": "sh_submit.sh"
}
def create_directories(ens: dict):
    for i in range(ens["ntraj"]):
        dirname = f"{ens['location']}/traj_{ens['idoffset']+i}"
        if os.path.isdir(dirname):
            os.system(f"rm -r {dirname}")
        os.mkdir(f"{dirname}") 

# REWORK
def copy_geom_files(ens: dict):
    for i in range(ens["ntraj"]):
        os.system(f"cp geom.xyz {ens['location']}/traj_{ens['idoffset']+i}/geom.xyz")

def create_subdirectories(ens: dict):
    for dir in get_dirs(f"{ens['location']}/"):
        if os.path.isdir(f"{ens['location']}/{dir}/backup"):
            os.system(f"rm -r {ens['location']}/{dir}/backup")
        os.mkdir(f"{ens['location']}/{dir}/backup")

        if os.path.isdir(f"{ens['location']}/{dir}/est"):
            os.system(f"rm -r {ens['location']}/{dir}/est")
        os.mkdir(f"{ens['location']}/{dir}/est")

        if os.path.isdir(f"{ens['location']}/{dir}/data"):
            os.system(f"rm -r {ens['location']}/{dir}/data")
        os.mkdir(f"{ens['location']}/{dir}/data")

def copy_input_files(ens: dict):
    for dir in get_dirs(f"{ens['location']}/"):
        os.system(f"cp {ens['input']} {ens['location']}/{dir}/input.inp")

def copy_submission_scripts(ens: dict):
    for dir in get_dirs(f"{ens['location']}/"):
        os.system(f"cp {ens['submission']} {ens['location']}/{dir}/submit.sh")

def copy_wavefunctions(ens: dict):
    if ens.get("wf") == "":
        return
    
    for dir in get_dirs(f"{ens['location']}/"):
        os.system(f"cp {ens['wf']} {ens['location']}/{dir}/est/wf.wf")

# ===========================================

def back_up_step(traj: Trajectory):
    """
    Creates a backup of the most recent step in the backup subdirectory.
    The whole traj object and the most recent WF are saved.
    These files are directly used for restarts.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    None
    """
    
    # save traj as pickle file
    with open("backup/traj.pkl", "wb") as traj_pkl:
        pickle.dump(traj, traj_pkl)

    # copy wavefunction if available
    if traj.est.program != "model":
        os.system(f"cp est/{traj.est.program}.wf backup/")

def write_headers(traj: Trajectory):
    """
    Creates all output files and writes their corresponding headers.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    None
    """
    
    # create log file and write its header
    # this file contains a record of all the stages of the calculation
    with open(traj.io.log_file, "w") as log_file:
        log_file.write(f"Calculation of {traj.par.name} dynamics\n\n")
    
    # create xyz file containing all the geometries
    with open(traj.io.xyz_file, "w") as xyz_file: pass

    # create h5 archive file
    # this file contains all the matrices at all steps
    with h5py.File(traj.io.mat_file, "w"): pass

    # create dat file and write its header
    # this file contains readily available quantities of the trajectory 
    with open(traj.io.dat_file, "w") as dat_file:
        dat_file.write(" ")
        # time
        dat_file.write(Printer.write('Time [fs]', "s"))
        # write header for each recorded quantity
        for record in traj.io.record:
            # active state
            if "active" == record:
                dat_file.write(Printer.write('Active State', "s"))
            # populations
            if "pop" == record:
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(f'{s+traj.est.skip} Population', "s"))
            # potential energy surfaces
            if "pes" == record:
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(f'{s+traj.est.skip} Pot En [eV]', "s"))
            # potential energy
            if "pen" == record:
                dat_file.write(Printer.write('Total Pot En [eV]', "s"))
            # total kinetic energy
            if "ken" == record:
                dat_file.write(Printer.write('Total Kin En [eV]', "s"))
            # total energy
            if "en" == record:
                dat_file.write(Printer.write('Total En [eV]', "s"))
            # nonadiabatic couplings
            if "nacdr" == record:
                for s1 in range(traj.par.n_states):
                    for s2 in range(s1):
                        dat_file.write(Printer.write(f'{s2+traj.est.skip}-{s1+traj.est.skip} NACdr [au]', "s"))
                        dat_file.write("Flip? ")
            # time-derivative couplings
            if "nacdt" == record:
                for s1 in range(traj.par.n_states):
                    for s2 in range(s1):
                        dat_file.write(Printer.write(f'{s2+traj.est.skip}-{s1+traj.est.skip} NACdt [au]', "s"))
            # wavefunction coefficient
            if "coeff" == record:
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(f'{s+traj.est.skip} State Coeff', f" <{Printer.field_length*2+1}"))
            # hopping probabilities
            if "prob" == record:
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(f'{s+traj.est.skip} Hopping Prob', "s"))
        dat_file.write("\n")

# should probably be reworked at some point
def time_log(traj: Trajectory, msg: str, *funcs):
    """
    Measures the time taken to execute the supplied functions.
    Writes the result to the log file.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    msg: str
        text preceding the time log
    
    Returns
    -------
    None
    
    Modifies
    --------
    ret: list
        all the returns of the functions supplied
    """
    
    with open(traj.io.log_file, "a") as log_file:
        # write the string
        log_file.write(Printer.write(msg, "s"))
        #start timer
        t0 = time.time()
        # catch function returns
        ret = []
        # execute all functions
        for func in funcs:
            ret.append(func())
        # stop timer
        t1 = time.time()
        log_file.write(f"{t1-t0:.4e}\n")
    return ret

def step_log(traj: Trajectory):
    """
    Write a header for a step to the log file.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    None
    """
    
    with open(traj.io.log_file, "a") as log_file:
        log_file.write(f"\nTime {Printer.write(traj.ctrl.curr_time*Constants.au2fs, '.4e')} fs, Step {Printer.write(traj.ctrl.curr_step, 'i')}\n")

def write_log(traj: Trajectory, msg: str):
    """
    Writes a string to the log file.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    msg: str
        string to be written (newline character has to be included)
    
    Returns
    -------
    None
    
    Modifies
    --------
    None
    """
    
    with open(traj.io.log_file, "a") as log_file:
        log_file.write(f"{msg}")

def write_xyz(traj: Trajectory):
    """
    Writes current geometry and velocities to the xyz file.
    Units are in AA and AA/fs respectively.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    None
    """
    
    with open(traj.io.xyz_file, "a") as xyz_file:
        xyz_file.write(f"{traj.par.n_atoms}\n")
        xyz_file.write(f"t = {traj.ctrl.curr_time*Constants.au2fs}\n")
        for a in range(traj.par.n_atoms):
            xyz_file.write(Printer.write(traj.geo_mn[-1,0].name_a[a], "s"))
            for i in range(3): xyz_file.write(Printer.write(traj.geo_mn[-1,0].position_ad[a,i]*Constants.bohr2A, "f"))
            for i in range(3): xyz_file.write(Printer.write(traj.geo_mn[-1,0].velocity_ad[a,i]*Constants.vau2vsi, "f"))
            xyz_file.write("\n")

def write_dat(traj: Trajectory):
    """
    Writes the requested data from the current step to the dat file.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    None
    """
    
    with open(traj.io.dat_file, "a") as dat_file:
        # time
        dat_file.write(Printer.write(traj.ctrl.curr_time*Constants.au2fs, "f"))
        # write the requested quantities in order
        for record in traj.io.record:
            # active state
            if record == "active":
                dat_file.write(Printer.write(traj.hop.active + traj.est.skip, "i"))
            # populations
            if record == "pop":
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(np.abs(traj.est.coeff_mns[-1,0,s])**2, "f"))
            # potential energy surfaces
            if record == "pes":
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(traj.pes_mn[-1,0].ham_diag_ss[s,s]*Constants.eh2ev, "f"))
            # potential energy
            if record == "pen":
                dat_file.write(Printer.write(traj.pes_mn[-1,0].poten*Constants.eh2ev, "f"))
            # total kinetic energy
            if record == "ken":
                dat_file.write(Printer.write(get_kinetic_energy(traj)*Constants.eh2ev, "f"))
            # total energy
            if record == "en":
                dat_file.write(Printer.write((get_kinetic_energy(traj) + traj.pes_mn[-1,0].poten)*Constants.eh2ev, "f"))
            # nonadiabatic couplings
            if record == "nacdr":
                for s1 in range(traj.par.n_states):
                    for s2 in range(s1):
                        nac = np.sum(traj.pes_mn[-1,0].nac_ddr_ssad[s1,s2]**2)
                        nac = np.sqrt(nac)
                        dat_file.write(Printer.write(nac, "f"))
                        dat_file.write(Printer.write(traj.pes_mn[-1,0].nac_flip_ss[s1,s2], "b"))
            # time-derivative couplings
            if record == "nacdt":
                for s1 in range(traj.par.n_states):
                    for s2 in range(s1):
                        dat_file.write(Printer.write(traj.pes_mn[-1,0].nac_ddt_ss[s1,s2], "f"))
            # WF coefficients
            if record == "coeff":
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(traj.est.coeff_mns[-1,0,s], "z"))
            # hopping probabilities
            if record == "prob":
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(traj.hop.prob_s[s], "f"))
        dat_file.write("\n")

def write_mat(traj: Trajectory):
    """
    Writes all the matrices of in the current step to the mat file.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    None
    """
    
    # compression format
    comp = "gzip"
    # compression level, max is 9
    comp_opt = 9
    with h5py.File(traj.io.mat_file, "a") as f:
        # group name is the current step id
        grp = f.create_group(f"{traj.ctrl.curr_step}")
        # current time
        grp.create_dataset("t", data=traj.ctrl.curr_time)

        # position
        grp.create_dataset("X", data=traj.geo_mn[-1,0].position_ad, compression=comp, compression_opts=comp_opt)
        # velocity
        grp.create_dataset("V", data=traj.geo_mn[-1,0].velocity_ad, compression=comp, compression_opts=comp_opt)
        # acceleration
        grp.create_dataset("A", data=traj.geo_mn[-1,0].force_ad, compression=comp, compression_opts=comp_opt)

        # diagonal hamiltonian
        grp.create_dataset("Hdiag", data=traj.pes_mn[-1,0].ham_diag_ss, compression=comp, compression_opts=comp_opt)
        # diabatic-diagonal unitary transformation
        grp.create_dataset("U", data=traj.pes_mn[-1,0].transform_ss, compression=comp, compression_opts=comp_opt)
        # nonadiabatic coupling
        grp.create_dataset("NACdr", data=traj.pes_mn[-1,0].nac_ddr_ssad, compression=comp, compression_opts=comp_opt)
        # time-derivative coupling
        grp.create_dataset("NACdt", data=traj.pes_mn[-1,0].nac_ddt_ss, compression=comp, compression_opts=comp_opt)

        # WF coefficients
        grp.create_dataset("C", data=traj.est.coeff_mns[-1,0], compression=comp, compression_opts=comp_opt)


def finalise_dynamics(traj: Trajectory):
    pass
