import os
import pickle
from re import search

from classes import TrajectorySH, Geometry, SimulationSH
from kinematics import *
from wigner import initial_sampling
from utility import get_dict_value, get_dirs
from constants import Constants

def create_submission_scripts(input_dict: dict):
    location = get_dict_value(input_dict["general"], "location", ".")
    script = get_dict_value(input_dict["general"], "submit", None)
    path = os.path.realpath(os.path.dirname(__file__))
    if script is None:
        script = "submit.sh"
        with open(script, "w") as file:
            file.write(f"export PYTHONPATH=$PYTHONPATH:{path}\n")
            file.write(f"python3 main.py input.inp")
    for dir in get_dirs(f"{location}/"):
        os.system(f"cp {location}/{script} {location}/{dir}")
        os.rename(f"{location}/{dir}/{script}", f"{location}/{dir}/submit.sh")
        os.system(f"cp {path}/main.py {location}/{dir}")

def create_directories(input_dict: dict):
    n_traj = int(get_dict_value(input_dict["ensemble"], "ntraj"))
    location = get_dict_value(input_dict["general"], "location", ".")
    id_offset = int(get_dict_value(input_dict["general"], "id_offset", "0"))
    
    for i in range(n_traj):
        dirname = f"{location}/traj_{id_offset+i}"
        if os.path.isdir(dirname):
            os.system(f"rm -r {dirname}")
        os.mkdir(f"{dirname}") 

def create_input_files(input_dict: dict):
    location = get_dict_value(input_dict["general"], "location", ".")

    for dir in get_dirs(f"{location}/"):
        with open(f"{location}/{dir}/input.inp", "w") as traj_input_file:
            for block in input_dict.keys():
                if block in ["ensemble"]:
                    continue
                
                traj_input_file.write(f"{block}\n")
                if block == "general":
                    traj_id = search(r"\d+", str(dir)).group()
                    traj_input_file.write(f"id={traj_id}\n")
                for key, value in input_dict[block].items():
                    traj_input_file.write(f"{key}={value}\n")
                traj_input_file.write("\n")

def create_geom_files(input_dict: dict):
    n_traj = int(get_dict_value(input_dict["ensemble"], "ntraj"))
    location = get_dict_value(input_dict["general"], "location", ".")
    geom_file = get_dict_value(input_dict["nuclear"], "input")
    id_offset = int(get_dict_value(input_dict["ensemble"], "id_offset", "0"))
    do_wigner = get_dict_value(input_dict["ensemble"], "wigner", "false") in Constants.true
    name = get_dict_value(input_dict["general"], "name", "name")

    init_geometry = Geometry(input_dict)
    positions, velocities = initial_sampling(n_traj, init_geometry, wigner=do_wigner)

    for i in range(n_traj):
        with open(f"{location}/traj_{id_offset+i}/{geom_file}", "w") as traj_xyz_file:
            traj_xyz_file.write(f"{init_geometry.n_atoms}\n")
            traj_xyz_file.write(f"{name}\n")
            for a in range(init_geometry.n_atoms):
                traj_xyz_file.write(f"{init_geometry.name_a[a]} ")
                traj_xyz_file.write(f"{positions[i,a,0]} {positions[i,a,1]} {positions[i,a,2]} ")
                traj_xyz_file.write(f"{velocities[i,a,0]} {velocities[i,a,1]} {velocities[i,a,2]}\n")

def create_subdirectories():
    if os.path.isdir("backup"):
        os.system("rm -r backup")
    os.mkdir("backup")

    if os.path.isdir("est"):
        os.system("rm -r est")
    os.mkdir("est")

    if os.path.isdir("data"):
        os.system("rm -r data")
    os.mkdir("data")

def back_up_step(traj: TrajectorySH, ctrl: SimulationSH):
    with open("backup/traj.pkl", "wb") as traj_pkl:
        pickle.dump(traj, traj_pkl)
    
    with open("backup/ctrl.pkl", "wb") as ctrl_pkl:
        pickle.dump(ctrl, ctrl_pkl)

def write_to_xyz(traj: TrajectorySH, ctrl: SimulationSH):
    with open(traj.io.xyz_file, "a") as xyz_file:
        xyz_file.write(f"{traj.geo.n_atoms}\n")
        xyz_file.write(f"t = {ctrl.current_time*Constants.au2fs}\n")
        for a in range(traj.geo.n_atoms):
            xyz_file.write(f"{traj.geo.name_a[a]} {traj.geo.position_ad[a,0]*Constants.bohr2A} \
            {traj.geo.position_ad[a,1]*Constants.bohr2A} {traj.geo.position_ad[a,2]*Constants.bohr2A}\n")
    
    with open(traj.io.dat_file, "a") as dat_file:
        dat_file.write(f"{ctrl.current_time*Constants.au2fs}")
        if "active_state" in traj.io.record:
            dat_file.write(f" {traj.hop.active}")
        if "population" in traj.io.record:
            outstr = ""
            for s in range(traj.est.n_states):
                outstr += f" {np.abs(traj.hop.state_coeff_s[s])**2} "
            dat_file.write(outstr)
        if "pes" in traj.io.record:
            dat_file.write(f" {traj.est.pes.ham_diag_ss.diagonal()}")
        if "total_energy" in traj.io.record:
            dat_file.write(f" {(get_kinetic_energy(traj) + traj.cons.potential_energy)*Constants.eh2ev}")
        if "nacme_norm" in traj.io.record:
            outstr = ""
            for s1 in range(traj.est.n_states):
                for s2 in range(s1):
                    nac = np.sum(traj.est.pes.nac_ddr_ssad[s1,s2,:,:] * traj.est.pes.nac_ddr_ssad[s1,s2,:,:])
                    nac = np.sqrt(nac)
                    outstr += f" {nac} {traj.est.pes.nac_flip[s1,s2]}"
            dat_file.write(outstr)
        if "state_coeff" in traj.io.record:
            dat_file.write(f" {traj.hop.state_coeff_s}")
        if "hopping_prob" in traj.io.record:
            dat_file.write(f" {traj.hop.prob_s}")
        if "com_cons" in traj.io.record:
            dat_file.write(f" {traj.cons.com_position} {traj.cons.com_velocity}")
        if "mom_cons" in traj.io.record:
            dat_file.write(f" {traj.cons.total_momentum + traj.cons.dmomentum - get_total_momentum(traj)}")
            dat_file.write(f" {traj.cons.total_ang_momentum + traj.cons.dang_momentum - get_total_ang_momentum(traj)}")
        dat_file.write("\n")

def finalise_dynamics(traj: TrajectorySH):
    pass