import numpy as np
import os, sys

from errors import *
from classes import Trajectory
from constants import Constants

from molpro_est import create_input_molpro, read_output_molpro_ham, read_output_molpro_nac, run_wfoverlap_molpro
from molcas_est import create_input_molcas_main, create_input_molcas_nac, read_output_molcas_ham, read_output_molcas_grad, read_output_molcas_nac, read_output_molcas_prop
from pyscf_est import run_pyscf_mcscf, run_pyscf_cisd
from turbomole_est import run_ricc2


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
    eval, evec = np.linalg.eigh(traj.pes.ham_diab_mnss[-1, 0])
    traj.pes.ham_transform_mnss[-1, 0] = evec
    traj.pes.ham_diag_mnss[-1, 0] = np.diag(eval)

def adjust_energy(traj: Trajectory):
    if traj.est.first:
        traj.par.ref_en = traj.pes.ham_diab_mnss[-1,0,0,0]
    for s in range(traj.par.n_states):
        traj.pes.ham_diab_mnss[-1, 0, s, s] -= traj.par.ref_en

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

def run_model(traj: Trajectory):
    traj.est.run(traj)
    diagonalise_hamiltonian(traj)

def run_molpro(traj: Trajectory):
    '''
    Runs molpro 202X calculation and reads the results
    modifies the trajectory class
    '''

    traj.est.file = f"{traj.est.program}"
    os.chdir("est")

    traj.est.calculate_nacs *= np.eye(traj.par.n_states)

    write_xyz(traj)
    create_input_molpro(traj.par.states, traj.est.file, traj.est.config, traj.est.calculate_nacs, traj.est.skip, False)

    err = os.system(f"molpro -W . -I . -d ./tmp_molpro -s {traj.est.file}.inp")
    if err > 0:
        print(f'Error in MOLPRO, exit code {err}')
        raise EstCalculationBrokenError

    ssum = np.cumsum(traj.par.states) - traj.par.states
    traj.pes.ham_diab_mnss[-1,0] = 0
    traj.pes.ham_diag_mnss[-1,0] = 0
    traj.pes.nac_ddr_mnssad[-1,0] = 0
    for s1, s2, i, j, val in read_output_molpro_ham(traj.est.file):
        if i < traj.par.n_states and j < traj.par.n_states:
            traj.pes.ham_diab_mnss[-1,0,i+ssum[s1],j+ssum[s2]] += val
    for s, i, j, a, val in read_output_molpro_nac(traj.est.file): traj.pes.nac_ddr_mnssad[-1,0,i+ssum[s],j+ssum[s],a] = val

    adjust_energy(traj)
    if traj.pes.diagonalise:
        diagonalise_hamiltonian(traj)

        # soc transformations, move later
        g_diab = np.zeros((traj.par.n_states, traj.par.n_states, traj.par.n_atoms, 3), dtype=np.complex128)
        for i in range(traj.par.n_states):
            for j in range(traj.par.n_states):
                g_diab[i,j] = (i == j)*traj.pes.nac_ddr_mnssad[-1,0,i,i] - (traj.pes.ham_diab_mnss[-1,0,i,i] - traj.pes.ham_diab_mnss[-1,0,j,j])*traj.pes.nac_ddr_mnssad[-1,0,i,j]
        g_diag = np.einsum("ij,jkad,kl->ilad", traj.pes.ham_transform_mnss[-1,0].conj().T, g_diab, traj.pes.ham_transform_mnss[-1,0])

        for i in range(traj.par.n_states):
            traj.pes.nac_ddr_mnssad[-1,0,i,i] = np.real(g_diag[i,i])

    else:
        traj.pes.ham_diag_mnss[-1, 0] = traj.pes.ham_diab_mnss[-1, 0]
        traj.pes.ham_transform_mnss[-1, 0] = np.identity(traj.par.n_states)

    overlap = True
    if overlap and traj.ctrl.substep == 0 and not traj.est.first:
        traj.pes.overlap_mnss[-1,0,:,:] = run_wfoverlap_molpro(traj.est.file, traj.geo.name_a, traj.geo.position_mnad[-1,0,:,:], traj.geo.position_mnad[-2,0,:,:], traj.est.config['basis'] , traj.par.n_states)

    adjust_nacmes(traj)

    os.chdir("..")
    traj.est.first = False


def run_turbo(traj: Trajectory):
    os.chdir('est')
    run_ricc2(traj, traj.est.config)
    os.chdir('..')
    if traj.pes.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes.ham_diag_mnss[-1, 0] = traj.pes.ham_diab_mnss[-1, 0]
        traj.pes.ham_transform_mnss[-1, 0] = np.identity(traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)



def run_pyscf_wrapper(traj: Trajectory):
    os.chdir('est')
    if traj.est.config['type'] == 'cisd':
        run_pyscf_cisd(traj, traj.est.config)
    else:
        run_pyscf_mcscf(traj, traj.est.config)
    os.chdir('..')
    if traj.pes.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes.ham_diag_mnss[-1, 0] = traj.pes.ham_diab_mnss[-1, 0]
        traj.pes.ham_transform_mnss[-1, 0] = np.identity(traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)

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


    traj.est.calculate_nacs *= np.eye(traj.par.n_states)
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
                    for i, i, a, val in read_output_molcas_grad(f"molcas.log", traj.est.config): traj.pes.nac_ddr_mnssad[-1,0,i-skip, i-skip, a] = val
                else:
                    for i, j, a, val in read_output_molcas_nac(f"molcas_{s2}_{s1}.log", s2, s1): traj.pes.nac_ddr_mnssad[-1,0,i-skip,j-skip,a] = val

    os.chdir("..")


    if traj.pes.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes.ham_diag_mnss[-1, 0] = traj.pes.ham_diab_mnss[-1, 0]
        traj.pes.ham_transform_mnss[-1, 0] = np.identity(traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)
