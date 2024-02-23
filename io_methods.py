import os
import pickle
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

def back_up_step(traj: Trajectory):
    with open("backup/traj.pkl", "wb") as traj_pkl:
        pickle.dump(traj, traj_pkl)

    if traj.est.program != "model":
        os.system("cp est/wf.wf backup/")

def write_headers(traj: Trajectory):
    with open(traj.io.log_file, "w") as log_file:
        log_file.write(f"Calculation of {traj.par.name} dynamics\n\n")
    
    with open(traj.io.xyz_file, "w") as xyz_file: pass

    with open(traj.io.dat_file, "w") as dat_file:
        dat_file.write(" ")
        dat_file.write(Printer.write('Time [fs]', "s"))
        for record in traj.io.record:
            if "active" == record:
                dat_file.write(Printer.write('Active State', "s"))
            if "pop" == record:
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(f'S{s+traj.est.skip} Population', "s"))
            if "pes" == record:
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(f'S{s+traj.est.skip} Pot En [eV]', "s"))
            if "pen" == record:
                dat_file.write(Printer.write('Total Pot En [eV]', "s"))
            if "ken" == record:
                dat_file.write(Printer.write('Total Kin En [eV]', "s"))
            if "en" == record:
                dat_file.write(Printer.write('Total En [eV]', "s"))
            if "nacme" == record:
                for s1 in range(traj.par.n_states):
                    for s2 in range(s1):
                        dat_file.write(Printer.write(f'S{s2+traj.est.skip}-S{s1+traj.est.skip} NAC [au]', "s"))
                        dat_file.write("Flip? ")
            if "coeff" == record:
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(f'S{s+traj.est.skip} State Coeff', f" <{Printer.field_length*2+1}"))
            if "prob" == record:
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(f'S{s+traj.est.skip} Hopping Prob', "s"))
        dat_file.write("\n")


def time_log(traj: Trajectory, msg: str, *funcs):
    with open(traj.io.log_file, "a") as log_file:
        log_file.write(Printer.write(msg, "s"))
        t0 = time.time()
        ret = []
        for func in funcs:
            ret.append(func())
        t1 = time.time()
        log_file.write(f"{t1-t0:.4e}\n")
    return ret

def step_log(traj: Trajectory):
    with open(traj.io.log_file, "a") as log_file:
        log_file.write(f"\nTime {Printer.write(traj.ctrl.curr_time*Constants.au2fs, '.4e')}, Step {Printer.write(traj.ctrl.curr_step, 'i')}\n")

def write_log(traj: Trajectory, msg: str):
    with open(traj.io.log_file, "a") as log_file:
        log_file.write(f"{msg}")

def write_xyz(traj: Trajectory):
    print(traj.ctrl.curr_time)
    
    with open(traj.io.xyz_file, "a") as xyz_file:
        xyz_file.write(f"{traj.par.n_atoms}\n")
        xyz_file.write(f"t = {traj.ctrl.curr_time*Constants.au2fs}\n")
        for a in range(traj.par.n_atoms):
            xyz_file.write(Printer.write(traj.geo.name_a[a], "s"))
            for i in range(3): xyz_file.write(Printer.write(traj.geo.position_mnad[-1,0,a,i]*Constants.bohr2A, "f"))
            for i in range(3): xyz_file.write(Printer.write(traj.geo.velocity_mnad[-1,0,a,i]*Constants.vau2vsi, "f"))
            xyz_file.write("\n")

def write_dat(traj: Trajectory):
    with open(traj.io.dat_file, "a") as dat_file:
        dat_file.write(Printer.write(traj.ctrl.curr_time*Constants.au2fs, "f"))
        for record in traj.io.record:
            if record == "active":
                dat_file.write(Printer.write(traj.hop.active + traj.est.skip, "i"))
            if record == "pop":
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(np.abs(traj.est.coeff_mns[-1,0,s])**2, "f"))
            if record == "pes":
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(traj.pes.ham_diag_mnss[-1,0,s,s]*Constants.eh2ev, "f"))
            if record == "pen":
                dat_file.write(Printer.write(traj.pes.ham_diag_mnss[-1,0,traj.hop.active,traj.hop.active]*Constants.eh2ev, "f"))
            if record == "ken":
                dat_file.write(Printer.write(get_kinetic_energy(traj)*Constants.eh2ev, "f"))
            if record == "en":
                dat_file.write(Printer.write((get_kinetic_energy(traj) + traj.pes.ham_diag_mnss[-1,0,traj.hop.active,traj.hop.active])*Constants.eh2ev, "f"))
            if record == "nacme":
                for s1 in range(traj.par.n_states):
                    for s2 in range(s1):
                        nac = np.sum(traj.pes.nac_ddr_mnssad[-1,0,s1,s2,:,:]**2)
                        nac = np.sqrt(nac)
                        dat_file.write(Printer.write(nac, "f")) 
                        dat_file.write(Printer.write(traj.pes.nac_flip[s1,s2], "b"))
            if record == "coeff":
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(traj.est.coeff_mns[-1,0,s], "z"))
            if record == "prob":
                for s in range(traj.par.n_states):
                    dat_file.write(Printer.write(traj.hop.prob_s[s], "f"))
        dat_file.write("\n")

def finalise_dynamics(traj: Trajectory):
    pass