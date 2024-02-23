import numpy as np
import os, sys

from errors import *
from classes import Trajectory
from constants import Constants
from io_methods import Printer

import fmodules.models_f as models_f

def set_est_sh(traj: Trajectory, nacs=True):
    if nacs:
        traj.est.calculate_nacs = np.ones_like(traj.est.calculate_nacs) - np.identity(traj.par.n_states)
    else:
        traj.est.calculate_nacs[:] = 0
    traj.est.calculate_nacs[traj.hop.active, traj.hop.active] = 1

def set_est_mfe(traj: Trajectory, nacs=True):
    if nacs:
        traj.est.calculate_nacs = np.ones_like(traj.est.calculate_nacs)
    else:
        traj.est.calculate_nacs = np.identity(traj.par.n_states)

def diagonalise_hamiltonian(traj: Trajectory):
    eval, evec = np.linalg.eigh(traj.pes.ham_diab_mnss[-1, traj.ctrl.substep])
    traj.pes.ham_transform_mnss[-1, traj.ctrl.substep] = evec
    traj.pes.ham_diag_mnss[-1, traj.ctrl.substep] = np.diag(eval)

def harm(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep,0,0] = 0.5*traj.geo.position_mnad[-1,traj.ctrl.substep,0,0]**2
    traj.pes.ham_diag_mnss[-1,traj.ctrl.substep] = traj.pes.ham_diab_mnss[-1,traj.ctrl.substep]
    traj.pes.nac_ddr_mnssad[-1,traj.ctrl.substep,0,0] = traj.geo.position_mnad[-1,traj.ctrl.substep]

def spin_boson(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep], gradH_ssad = models_f.spin_boson(traj.geo.position_mnad[-1, traj.ctrl.substep])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, traj.ctrl.substep] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, traj.ctrl.substep], traj.pes.ham_transform_mnss[-1, traj.ctrl.substep], gradH_ssad)
    #adjust_nacmes(traj)

def tully_1(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep], gradH_ssad = models_f.tully_1(traj.geo.position_mnad[-1, traj.ctrl.substep])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, traj.ctrl.substep] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, traj.ctrl.substep], traj.pes.ham_transform_mnss[-1, traj.ctrl.substep], gradH_ssad)
    #adjust_nacmes(traj)

def tully_s(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep], gradH_ssad = models_f.tully_s(traj.geo.position_mnad[-1, traj.ctrl.substep])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, traj.ctrl.substep] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, traj.ctrl.substep], traj.pes.ham_transform_mnss[-1, traj.ctrl.substep], gradH_ssad)
    adjust_nacmes(traj)

def tully_n(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep], gradH_ssad = models_f.tully_n(traj.geo.position_mnad[-1, traj.ctrl.substep])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, traj.ctrl.substep] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, traj.ctrl.substep], traj.pes.ham_transform_mnss[-1, traj.ctrl.substep], gradH_ssad)
    adjust_nacmes(traj)

def tully_2(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep], gradH_ssad = models_f.tully_2(traj.geo.position_mnad[-1, traj.ctrl.substep])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, traj.ctrl.substep] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, traj.ctrl.substep], traj.pes.ham_transform_mnss[-1, traj.ctrl.substep], gradH_ssad)
    adjust_nacmes(traj)

def tully_3(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep], gradH_ssad = models_f.tully_3(traj.geo.position_mnad[-1, traj.ctrl.substep])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, traj.ctrl.substep] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, traj.ctrl.substep], traj.pes.ham_transform_mnss[-1, traj.ctrl.substep], gradH_ssad)
    adjust_nacmes(traj)

def sub_x(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep], gradH_ssad = models_f.sub_x(traj.geo.position_mnad[-1, traj.ctrl.substep])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, traj.ctrl.substep] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, traj.ctrl.substep], traj.pes.ham_transform_mnss[-1, traj.ctrl.substep], gradH_ssad)
    adjust_nacmes(traj)

def sub_s(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep], gradH_ssad = models_f.sub_s(traj.geo.position_mnad[-1, traj.ctrl.substep])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, traj.ctrl.substep] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, traj.ctrl.substep], traj.pes.ham_transform_mnss[-1, traj.ctrl.substep], gradH_ssad)
    adjust_nacmes(traj)

def sub_2(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,traj.ctrl.substep], gradH_ssad = models_f.sub_2(traj.geo.position_mnad[-1, traj.ctrl.substep])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, traj.ctrl.substep] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, traj.ctrl.substep], traj.pes.ham_transform_mnss[-1, traj.ctrl.substep], gradH_ssad)
    adjust_nacmes(traj)

def lvc_wrapper(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], traj.pes.ham_diag_mnss[-1,0], traj.pes.ham_transform_mnss[-1,0], traj.pes.nac_ddr_mnssad[-1,0] = \
        LVC.get_est(traj.geo.position_mnad[-1,0].flatten())

def run_molpro(traj: Trajectory):
    '''
    Runs molpro 202X calculation and reads the results
    modifies the trajectory class
    '''

    #traj.est.file = f"{traj.est.program}_{traj.ctrl.curr_step}_{traj.ctrl.substep}"
    traj.est.file = f"{traj.est.program}"
    os.chdir("est")

    write_xyz(traj)
    create_input_molpro(traj.est.file, traj.est.config, traj.est.calculate_nacs, traj.est.skip, False)

    err = os.system(f"molpro -W . -I . -d ./tmp_molpro -s {traj.est.file}.inp")
    if err > 0:
        print(f'Error in MOLPRO, exit code {err}')
        raise EstCalculationBrokenError

    if traj.est.first:
        with open("energies.dat", "w") as f:
            f.write(Printer.write(" Time [fs]", "s"))
            for i in range(traj.est.config["sa"]):
                f.write(Printer.write(f"S{i} Energy [eV]", "s"))
            f.write("\n")

    skip = traj.est.skip
    with open("energies.dat", "a") as f:
        f.write(Printer.write(traj.ctrl.curr_time*Constants.au2fs, "f"))
        for i, j, val in read_output_molpro_ham(traj.est.file):
            f.write(Printer.write(val*Constants.eh2ev, "f"))
            if i-skip >= 0 and j-skip >= 0 and i-skip < traj.par.n_states and j-skip < traj.par.n_states:
                traj.pes.ham_diab_mnss[-1,0,i-skip,j-skip] = val
        f.write("\n")
        
    for i, j, a, val in read_output_molpro_nac(traj.est.file): traj.pes.nac_ddr_mnssad[-1,0,i-skip,j-skip,a] = val

    os.chdir("..")

    if traj.pes.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes.ham_diag_mnss[-1, traj.ctrl.substep] = traj.pes.ham_diab_mnss[-1, traj.ctrl.substep]
        traj.pes.ham_transform_mnss[-1, traj.ctrl.substep] = np.identity(traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)

def interpolate_est(traj: Trajectory, x1):
    os.remove("est/wf.wf")
    os.system("cp backup/wf.wf est/")

    x0 = traj.geo.position_mnad[-2,0]
    n = 2
    for i in range(n):
        traj.geo.position_mnad[-1,0] = ((n-i)*x0 + i*x1)/n
        traj.est.run(traj)
    traj.geo.force_mnad[-1,0] = -traj.pes.nac_ddr_mnssad[-1,0,traj.hop.active,traj.hop.active]/traj.geo.mass_a[:,None]

def adjust_energy(traj: Trajectory):
    if traj.est.first:
        traj.par.ref_en = traj.pes.ham_diag_mnss[-1,0,0,0]
        traj.est.first = False
    for s in range(traj.par.n_states):
        traj.pes.ham_diag_mnss[-1, traj.ctrl.substep, s, s] -= traj.par.ref_en


def adjust_nacmes(traj: Trajectory):
    '''
    Flips nacmes by looking at simple overlap between them
    Should check to ensure that this works with actual wf/ci overlaps
    '''
    for s1 in range(traj.par.n_states):
        for s2 in range(s1+1, traj.par.n_states):
            if np.sum(traj.pes.nac_ddr_mnssad[-2, 0, s1, s2, :, :]*traj.pes.nac_ddr_mnssad[-1, 0, s1, s2, :, :]) < 0:
                traj.pes.nac_ddr_mnssad[-1, 0, s1, s2, :, :] = -traj.pes.nac_ddr_mnssad[-1, 0, s1, s2, :, :]
                traj.pes.nac_flip[s1,s2] = True
                traj.pes.nac_flip[s2,s1] = True
            else:
                traj.pes.nac_flip[s1,s2] = False
                traj.pes.nac_flip[s2,s1] = False
            traj.pes.nac_ddr_mnssad[-1, 0, s2, s1, :, :] = -traj.pes.nac_ddr_mnssad[-1, 0, s1, s2, :, :]

def write_xyz(traj: Trajectory):
    with open(f"{traj.est.file}.xyz", "w") as file:
        file.write(f"{traj.par.n_atoms}\n\n")
        for a in range(traj.par.n_atoms): 
            file.write(f"{traj.geo.name_a[a]} \
                         {traj.geo.position_mnad[-1,0,a,0]*Constants.bohr2A} \
                         {traj.geo.position_mnad[-1,0,a,1]*Constants.bohr2A} \
                         {traj.geo.position_mnad[-1,0,a,2]*Constants.bohr2A}\n")


def create_input_molpro(file_root: str, config: dict, calculate_nacs: np.ndarray, skip: int, mld: bool = False):
    n_states = calculate_nacs.shape[0]

    file = f"{file_root}.inp"
    with open(file, "w") as f:
        #File and threshold section
        f.write(f"***\n")
        f.write(f"file,2,wf.wf\n")
        f.write("memory,100,m\n")
        f.write("gprint,orbitals,civector,angles=-1,distance=-1\n")
        f.write(" gthresh,twoint=1.0d-13\n")
        f.write(" gthresh,energy=1.0d-10,gradient=1.0d-10\n") # TODO add thresholds...
        f.write(" gthresh,printci=0.000000009,thrprint=0\n")
        f.write(f"punch,{file_root}.pun,new\n")

        # Basis and geometry section
        f.write(f"basis={config['basis']}\n")
        f.write("symmetry,nosym;\n")
        f.write("orient,noorient;\n")
        f.write("geomtype=xyz;\n")
        f.write(f"geom={file_root}.xyz\n")

        #mcscf section - with second order if needed
        # MAX 10 CPMCSCF CALLS IN ONE MULTI
        if config["df"]:
            f.write("{df-multi," + f"df_basis={config['dfbasis']}," + "so;\n")
        else:
            f.write("{multi, so;\n")
        f.write("maxiter,40;\n")
        f.write(f"occ,{config['active']};\n")
        f.write(f"closed,{config['closed']};\n")
        f.write(f"wf,{config['nel']},1,0;\n")
        f.write(f"state,{config['sa']};\n")
        f.write("print, orbitals;")

        record = 5100.1
        for i in range(n_states):
            if calculate_nacs[i, i]:
                f.write(f"CPMCSCF,GRAD,{i+skip+1}.1,accu=1.0d-12,record={record};\n")
                record += 1

        for i in range(n_states):
            for j in range(i + 1, n_states):
                if calculate_nacs[i, j]:
                    f.write(f"CPMCSCF,NACM,{i+skip+1}.1,{j+skip+1}.1,accu=1.0d-12,record={record};\n")
                    record += 1
        f.write("}\n")

        record = 5100.1
        for i in range(n_states):
            if calculate_nacs[i, i]:
                f.write(f"{{FORCES;SAMC,{record}}};\n")
                record += 1

        for i in range(n_states):
            for j in range(i + 1, n_states):
                if calculate_nacs[i, j]:
                    f.write(f"{{FORCES;SAMC,{record}}};\n")
                    record += 1

        if mld:
            f.write(f"put,molden, {file_root}.mld\n")
        f.write("---")

def read_output_molpro_ham(file_root: str):
    with open(f"{file_root}.out", "r") as file:
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
                yield state, state, float(data[-1])

            #SOC ham elements read here

def read_output_molpro_nac(file_root: str):
    with open(f"{file_root}.out", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break

            line = line.strip().lower()

            if line.startswith("sa-mc gradient for"):
                state = int(line.split()[-1].split(".")[0]) - 1
                for i in range(3): file.readline()
                
                a = 0
                while (line := file.readline().strip()):
                    data = line.strip().split()
                    yield state, state, a, [float(x) for x in data[1:]]
                    a += 1

            if line.strip().startswith("sa-mc nacme for"):
                state1 = int(line.split()[-3].split(".")[0]) - 1
                state2 = int(line.split()[-1].split(".")[0]) - 1
                for i in range(3): file.readline()
                
                a = 0
                while (line := file.readline().strip()):
                    data = line.strip().split()
                    yield state1, state2, a, [float(x) for x in data[1:]]
                    yield state2, state1, a, [-float(x) for x in data[1:]]
                    a += 1

"""
Created by Joseph Charles Cooper
"""
# write RC file
def run_molcas(traj: Trajectory):
    '''
    Runs and reads molcas output file
    '''

    os.chdir("est")
    skip = traj.est.skip


    write_xyz(traj)
    create_input_molcas_main(traj.est.file, traj.est.config, traj.est.calculate_nacs, skip)
    for i in range(traj.par.n_states):
        for j in range(i + 1, traj.par.n_states):
            if traj.est.calculate_nacs[i, j]:
                create_input_molcas_nac(traj.est.file, traj.est.config, traj.est.calculate_nacs, skip, i, j)

    err = os.system(f"pymolcas -f -b1 molcas.input")
    if err:
        print(f'Error code {err} in est')
        raise EstCalculationBrokenError
    for i in range(traj.par.n_states):
        for j in range(i + 1, traj.par.n_states):
            if traj.est.calculate_nacs[i, j]:
                err = os.system(f"pymolcas -f -b1 molcas_{i}_{j}.input")
                if err:
                    print(f"Error code {err} in est")
                    raise EstCalculationBrokenError

    for i, j, val in read_output_molcas_ham('molcas.log', traj.est.config):
        if i-skip >= 0 and j-skip >= 0 and i-skip < traj.par.n_states and j-skip < traj.par.n_states:
            traj.pes.ham_diab_mnss[-1,0,i-skip,j-skip] = val
    for s1 in range(traj.par.n_states):
        for s2 in range(i + 1, traj.par.n_states):
            if traj.est.calculate_nacs[i, j]:
                for i, j, a, val in read_output_molcas_nac(f"molcas_{i}_{j}.log", i, j): traj.pes.nac_ddr_mnssad[-1,0,i-skip,j-skip,a] = val

    os.chdir("..")

    if traj.pes.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes.ham_diag_mnss[-1, traj.ctrl.substep] = traj.pes.ham_diab_mnss[-1, traj.ctrl.substep]
        traj.pes.ham_transform_mnss[-1, traj.ctrl.substep] = np.identity(traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)

def create_input_molcas_main(file_root: str, config: dict, calculate_nacs: np.ndarray, skip: int):
    n_states = calculate_nacs.shape[0]

    with open(f"{file_root}.input", "w") as file:
        file.write("&GATEWAY\n")
        file.write(f"XYZ={file_root}.xyz\n")
        #  if SOC:
            #  f.write("AMFI")
        # Need to change to angstrom

        file.write("GROUP=NOSYM\n")
        file.write("RICD\n")
        file.write(f"BASIS={config['basis']}\n\n")
        file.write("&SEWARD\n\n")
        file.write("&RASSCF\n")
        file.write("JOBIPH\nCIRESTART\n")
        nactel = config['nel'] - 2*config['closed']
        file.write(f"NACTEL={nactel} 0 0\n")
        ras2 = config['active'] - config['closed']
        file.write(f"RAS2={ras2}\n")
        file.write(f"INACTIVE={config['closed']}\n")
        for i in range(n_states):
            if calculate_nacs[i, i]:
                file.write(f"RLXROOT={i+skip+1}\n")
        file.write(f"CIROOT={n_states} {n_states} 1\n\n")

        file.write(">>> COPY JOB001 JOB002\n")
        file.write(">>> COPY molcas.JobIph JOB001\n")
        file.write(">>> COPY molcas.JobIph JOBOLD\n")

        if config.get("caspt2", False):
            file.write(f"&CASPT2\n")
            file.write(f"GRDT\n")
            file.write(f"imag={config['imag']}\n")
            file.write(f"shift={config['shift']}\n")
            file.write("convergence = 1e-8\n")
            file.write(f"ipea=0.0\n")
            if config["caspt2_type"].upper()[0] == 'R':
                file.write(f"RMUL=all\n\n")
            elif config["caspt2_type"].upper()[0] == 'M':
                file.write(f"MULT=all\n\n")
            else:
                file.write(f"XMUL=all\n\n")
            file.write(">>> COPY molcas.JobMix JOB001\n\n")

        file.write(f"&ALASKA\n\n")

        file.write(f"&RASSI\n")

        if config.get("caspt2", False):
            file.write("EJOB\n")
        #  if SOC:
            #  f.write("SPIN\nSOCO=0\n")
        file.write("DIPR = -1\n")
        file.write("STOVERLAPS\n")


def create_input_molcas_nac(file_root: str, config: dict, calculate_nacs: np.ndarray, skip: int, nacidx1: int, nacidx2: int):
    n_states = calculate_nacs.shape[0]

    with open(f"{file_root}_{nacidx1}_{nacidx2}.input", "w") as file:
        file.write("&GATEWAY\n")
        file.write(f"XYZ={file_root}.xyz\n")
        #  if SOC:
            #  f.write("AMFI")
        # Need to change to angstrom

        file.write("GROUP=NOSYM\n")
        file.write("RICD\n")
        file.write(f"BASIS={config['basis']}\n\n")
        file.write("&SEWARD\n\n")
        file.write("&RASSCF\n")
        file.write("JOBIPH\nCIRESTART\n")
        nactel = config['nel'] - 2*config['closed']
        file.write(f"NACTEL={nactel} 0 0\n")
        ras2 = config['active'] - config['closed']
        file.write(f"RAS2={ras2}\n")
        file.write(f"INACTIVE={config['closed']}\n")
        for i in range(n_states):
            if calculate_nacs[i, i]:
                file.write(f"RLXROOT={i+skip+1}\n")
        file.write(f"CIROOT={n_states} {n_states} 1\n\n")

        file.write(">>> COPY molcas.JobIph JOB001\n")

        if config.get("caspt2", False):
            file.write(f"&CASPT2\n")
            file.write(f"GRDT\n")
            if nacidx1 != nacidx2:
                file.write(f"NAC = {nacidx1+skip+1} {nacidx2+skip+1}\n")
            file.write("convergence = 1e-8\n")
            file.write(f"imag={config['imag']}\n")
            file.write(f"shift={config['shift']}\n")
            file.write(f"ipea=0.0\n")
            if config["caspt2_type"].upper()[0] == 'R':
                file.write(f"RMUL=all\n\n")
            elif config["caspt2_type"].upper()[0] == 'M':
                file.write(f"MULT=all\n\n")
            else:
                file.write(f"XMUL=all\n\n")
            file.write(">>> COPY molcas.JobMix JOB001\n\n")

        file.write(f"&ALASKA\n")
        if nacidx1 != nacidx2:
            file.write(f"NAC = {nacidx1+skip+1} {nacidx2+skip+1}\n")


def read_output_molcas_ham(file_name: str, config: dict):
    with open(file_name, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            line = line.strip().lower()

            if False: #SOC
                pass
            else:
                if config.get("caspt2", False):
                    if 'ms-caspt2 energies' in line:
                        while (line := file.readline()):
                            data = line.strip().split()
                            yield int(data[-4]), float(data[-1])

                else:
                    if 'final state energy(ies)' in line:
                        for i in range(2): file.readline()
                        while (line := file.readline()):
                            data = line.strip().split()
                            yield int(data[-4]), int(data[-4]), float(data[-1])

def read_output_molcas_grad(file_name: str, config: dict):
    with open(file_name, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            line = line.strip().lower()

            if 'RLXROOT' in line:
                state = int(line.split('=')[-1]) - 1

            if 'Molecular gradients' in line:
                for i in range(7): file.readline()
                
                a = 0
                while (line := file.readline()):
                    data = line.strip().split()
                    yield state, state, a, [float(x) for x in data[1:]]

def read_output_molcas_nac(file_name, i, j):
    with open(file_name, 'r') as file:
        if i != j:
            state1 = i
            state2 = j
        else:
            state1 = i
            state2 = i

        file.seek(0)
        while True:
            line = file.readline()
            if not line:
                break

            line = line.strip().lower()
            if 'Total derivative coupling' in line or 'Molecular gradients' in line:
                for i in range(7): file.readline()

                a = 0
                while (line := file.readline()):
                    data = line.strip().split()
                    yield state1, state2, a, [float(x) for x in data[1:]]
                    yield state2, state1, a, [-float(x) for x in data[1:]]
