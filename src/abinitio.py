import numpy as np
import os
from typing import TextIO

from constants import Constants
from classes import TrajectorySH, SimulationSH

import fortran_modules.models_f as models_f

def diagonalise_hamiltonian(traj: TrajectorySH):
    eval, evec = np.linalg.eigh(traj.est.pes.ham_diab_ss)
    traj.est.pes.ham_transform_ss = evec
    for s in range(traj.est.n_states):
        traj.est.pes.ham_diag_ss[s,s] = eval[s]

def get_electronic_structure(traj: TrajectorySH, ctrl: SimulationSH):
    if traj.est.type.startswith("model"):
        if "sb_1" in traj.est.type:
            traj.est.pes.ham_diab_ss, gradH_ssad = models_f.spin_boson(traj.geo.position_ad)
        elif "tully_1" in traj.est.type:
            #traj.H_ss, traj.grad_ssad = models.tully_1(traj.position_ad)
            traj.est.pes.ham_diab_ss, gradH_ssad = models_f.tully_1(traj.geo.position_ad)
        elif "tully_2" in traj.est.type:
            traj.est.pes.ham_diab_ss, gradH_ssad = models_f.tully_2(traj.geo.position_ad)
        elif "tully_3" in traj.est.type:
            traj.est.pes.ham_diab_ss, gradH_ssad = models_f.tully_3(traj.geo.position_ad)
        diagonalise_hamiltonian(traj)
        traj.est.pes.nac_ddr_ssad = models_f.get_nac_and_gradient(traj.est.pes.ham_diag_ss, traj.est.pes.ham_transform_ss, gradH_ssad)
    else:
        inp_out(traj, ctrl)
        if traj.est.pes.diagonalise:
            diagonalise_hamiltonian(traj)
        else:
            traj.est.pes.ham_diag_ss = traj.est.pes.ham_diab_ss
            traj.est.pes.ham_transform_ss = np.identity(traj.est.n_states)
        
    adjust_energy(traj)
    adjust_nacmes(traj)

    for s in range(traj.est.n_states):
        traj.geo.force_sad[s,:,:] = -traj.est.pes.nac_ddr_ssad[s,s,:,:]
    traj.cons.potential_energy = traj.est.pes.ham_diag_ss[traj.hop.active, traj.hop.active]

def inp_out(traj: TrajectorySH, ctrl: SimulationSH):

    os.chdir("est")
    create_input(traj, ctrl)
    os.system(f"molpro -W . -I . -d ./tmp_molpro -s {traj.est.file}{ctrl.step}.inp")
    
    while not os.path.isfile(f"{traj.est.file}{ctrl.step}.out"):
        pass

    read_output_molpro(traj, f"{traj.est.file}{ctrl.step}.out")
    os.chdir("..")
    
def adjust_energy(traj: TrajectorySH):
    if traj.est.first:
        traj.est.reference_energy = traj.est.pes.ham_diag_ss[0,0]
    for s in range(traj.est.n_states):
        traj.est.pes.ham_diag_ss[s,s] -= traj.est.reference_energy

def adjust_nacmes(traj: TrajectorySH):
    for s1 in range(traj.est.n_states):
        for s2 in range(s1+1, traj.est.n_states):
            if np.sum(traj.est.pes_old.nac_ddr_ssad[s1,s2,:,:]*traj.est.pes.nac_ddr_ssad[s1,s2,:,:]) < 0:
                traj.est.pes.nac_ddr_ssad[s1,s2,:,:] = -traj.est.pes.nac_ddr_ssad[s1,s2,:,:]
                traj.est.pes.nac_flip[s1,s2] = True
                traj.est.pes.nac_flip[s2,s1] = True
            else:
                traj.est.pes.nac_flip[s1,s2] = False
                traj.est.pes.nac_flip[s2,s1] = False
            traj.est.pes.nac_ddr_ssad[s2,s1,:,:] = -traj.est.pes.nac_ddr_ssad[s1,s2,:,:]

def create_input(traj: TrajectorySH, ctrl: SimulationSH):
    # Molpro input, it must be changed for other codes
    with open(f"{traj.est.file}{ctrl.step}.inp", "w") as f:
        f.write(f"***, {traj.name} calculation of {ctrl.step} step in trajectory {traj.id}\n")
        f.write(f"file,2,wf.wf\n")
        f.write("memory,100,m\n")
        f.write("gprint,orbitals,civector,angles=-1,distance=-1\n")
        f.write(" gthresh,twoint=1.0d-13\n")
        f.write(" gthresh,energy=1.0d-7,gradient=1.0d-2\n")
        f.write(" gthresh,thrpun=0.001\n")
        f.write(f"punch,{traj.est.file}{ctrl.step}.pun,new\n")
        
        f.write(f"basis={traj.est.basis}\n")
        f.write("symmetry,nosym;\n")
        f.write("orient,noorient;\n")
        f.write("bohr;\n")
        f.write("geomtype=xyz;\n")
        f.write("geom={"+"\n")
        f.write(f"{traj.geo.n_atoms}\n")
        f.write("\n")
        for a in range(traj.geo.n_atoms):
            line = f"{traj.geo.name_a[a]} {traj.geo.position_ad[a,0]} {traj.geo.position_ad[a,1]} {traj.geo.position_ad[a,2]}\n"
            f.write(line)
        f.write("}\n")
        
        f.write("{" + "df-"*traj.est.density_fit + "multi," + "df_basis=avdz,"*traj.est.density_fit + "so;\n")
        f.write("maxiter,40;\n")
        f.write(f"occ,{traj.est.active_orb};\n")
        f.write(f"closed,{traj.est.closed_orb};\n")
        f.write(f"wf,{traj.est.n_el},1,0;\n")
        f.write(f"state,{traj.est.n_states};\n")

        record = 5100.1
        f.write(f"CPMCSCF,GRAD,{traj.hop.active+1}.1,record={record};\n")
        record += 1

        if not traj.est.recalculate:
            for i in range(traj.est.n_states):
                for j in range(i+1, traj.est.n_states):
                    f.write(f"CPMCSCF,NACM,{i+1}.1,{j+1}.1,record={record};\n")
                    record += 1
            f.write("}\n")

        record = 5100.1
        f.write(f"{{FORCES;SAMC,{record}}};\n")
        record += 1

        if not traj.est.recalculate:
            for i in range(traj.est.n_states):
                for j in range(i+1, traj.est.n_states):
                    f.write(f"{{FORCES;SAMC,{record}}};\n")
                    record += 1

        #f.write(f"put,molden, {traj.est.file}{ctrl.step}.mld\n")
        f.write("---")

# re-implement using yield
def read_output_molpro(traj: TrajectorySH, filename: str):
    with open(filename, "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            
            line = line.strip().lower()
                
            if line.startswith("results for state "):
                state = int(line.split()[-1].split(".")[0]) - 1
                file.readline()
                line = file.readline().strip()
                data = line.split()
                traj.est.pes.ham_diab_ss[state, state] = float(data[-1])

            if line.startswith("sa-mc gradient for"):
                state = int(line.split()[-1].split(".")[0]) - 1
                at = 0
                for i in range(3): file.readline()
                for at in range(traj.geo.n_atoms):
                    line = file.readline().strip()
                    data = line.split()
                    traj.est.pes.nac_ddr_ssad[state, state, at, :] = [float(x) for x in data[1:]]
            
            if line.strip().startswith("sa-mc nacme for"):
                state1 = int(line.split()[-3].split(".")[0]) - 1
                state2 = int(line.split()[-1].split(".")[0]) - 1
                at = 0
                for i in range(3): file.readline()
                for at in range(traj.geo.n_atoms):
                    line = file.readline().strip()
                    data = line.split()
                    traj.est.pes.nac_ddr_ssad[state1, state2, at, :] = [float(x) for x in data[1:]]
                    traj.est.pes.nac_ddr_ssad[state2, state1, at, :] = traj.est.pes.nac_ddr_ssad[state1, state2, at, :]