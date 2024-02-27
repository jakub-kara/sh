import numpy as np
import os, sys

from errors import *
from classes import Trajectory
from constants import Constants

import fmodules.models_f as models_f

from molpro_est import create_input_molpro, read_output_molpro_ham, read_output_molpro_nac, run_wfoverlap_molpro
from molcas_est import create_input_molcas_main, create_input_molcas_nac, read_output_molcas_ham, read_output_molcas_grad, read_output_molcas_nac, read_output_molcas_prop


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




def run_molpro(traj: Trajectory):
    '''
    Runs molpro 202X calculation and reads the results
    modifies the trajectory class
    '''

    #traj.est.file = f"{traj.est.program}_{traj.ctrl.curr_step}_{traj.ctrl.substep}"
    traj.est.file = f"{traj.est.program}"
    os.chdir("est")

    #  traj.est.calculate_nacs *= np.eye(traj.par.n_states)

    write_xyz(traj)
    create_input_molpro(traj.est.file, traj.est.config, traj.est.calculate_nacs, traj.est.skip, False)

    err = os.system(f"molpro -W . -I . -d ./tmp_molpro -s {traj.est.file}.inp")
    if err > 0:
        print(f'Error in MOLPRO, exit code {err}')
        raise EstCalculationBrokenError

    skip = traj.est.skip
    for i, j, val in read_output_molpro_ham(traj.est.file): 
        if i-skip >= 0 and j-skip >= 0 and i-skip < traj.par.n_states and j-skip < traj.par.n_states:
            traj.pes.ham_diab_mnss[-1,0,i-skip,j-skip] = val
    for i, j, a, val in read_output_molpro_nac(traj.est.file): traj.pes.nac_ddr_mnssad[-1,0,i-skip,j-skip,a] = val


    if traj.pes.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes.ham_diag_mnss[-1, traj.ctrl.substep] = traj.pes.ham_diab_mnss[-1, traj.ctrl.substep]
        traj.pes.ham_transform_mnss[-1, traj.ctrl.substep] = np.identity(traj.par.n_states)

    overlap = True
    if overlap and not traj.est.first:
        traj.pes.overlap_mnss[-1,0,:,:] = run_wfoverlap_molpro(traj.est.file, traj.geo.name_a, traj.geo.position_mnad[-1,0,:,:], traj.geo.position_mnad[-2,0,:,:], traj.est.config['basis'] , traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)


    os.chdir("..")
        


"""
Created by Joseph Charles Cooper
"""
# write RC file
def run_molcas(traj: Trajectory):
    '''
    Runs and reads molcas output file
    '''
    traj.est.file = f"{traj.est.program}"

    os.chdir("est")
    skip = traj.est.skip


    #  traj.est.calculate_nacs *= np.eye(traj.par.n_states)
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

    traj.pes.overlap_mnss[-1,0,:,:] = read_output_molcas_prop('molcas.log', traj.est.config)

    for s1 in range(traj.par.n_states):
        for s2 in range(s1+1):
            if traj.est.calculate_nacs[s1, s2]:
                if s1 == s2:
                    print(s1,s2)
                    for i, i, a, val in read_output_molcas_grad(f"molcas.log", traj.est.config): traj.pes.nac_ddr_mnssad[-1,0,i-skip, i-skip, a] = val
                else:
                    for i, j, a, val in read_output_molcas_nac(f"molcas_{s2}_{s1}.log", s2, s1): traj.pes.nac_ddr_mnssad[-1,0,i-skip,j-skip,a] = val

    os.chdir("..")


    if traj.pes.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes.ham_diag_mnss[-1, traj.ctrl.substep] = traj.pes.ham_diab_mnss[-1, traj.ctrl.substep]
        traj.pes.ham_transform_mnss[-1, traj.ctrl.substep] = np.identity(traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)

