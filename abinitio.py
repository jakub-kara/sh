import numpy as np
import os, sys

from errors import *
from classes import Trajectory
from constants import Constants

from molpro_est import create_input_molpro, read_output_molpro_ham, read_output_molpro_nac, run_wfoverlap_molpro, read_output_molpro_dip
from molcas_est import create_input_molcas_main, create_input_molcas_nac, read_output_molcas_ham, read_output_molcas_grad, read_output_molcas_nac, read_output_molcas_prop
from pyscf_est import run_pyscf_mcscf, run_pyscf_cisd
from turbomole_est import run_ricc2


def request_nacs_sh(traj: Trajectory, nacs=True):
    """
    Sets boolean array to indicate which gradients and nacmes should be calculated in EST.
    SH-specific.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    nacs: bool
        Flags whether nacmes should be requested
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.est.calc_nacs
    """
    
    # 0s on diagonal, 1s off diagonal
    if nacs: traj.est.calc_nacs = np.ones_like(traj.est.calc_nacs) - np.identity(traj.par.n_states)
    # 0s everywhere
    else: traj.est.calc_nacs[:] = 0
    # request gradient for active state
    traj.est.calc_nacs[traj.hop.active, traj.hop.active] = 1

def request_nacs_mfe(traj: Trajectory, nacs=True):
    """
    Sets boolean array to indicate which gradients and nacmes should be calculated in EST.
    MFE-specific.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    nacs: bool
        Flags whether nacmes should be requested
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.est.calc_nacs
    """
    
    # request all gradients and nacmes
    if nacs: traj.est.calc_nacs = np.ones_like(traj.est.calc_nacs)
    # request all gradients
    else: traj.est.calc_nacs = np.identity(traj.par.n_states)

def diagonalise_hamiltonian(traj: Trajectory):
    """
    Takes the latest diabatic Hamiltonian and performs diagonalisation.
    Resulting diagonal Hamiltonian and unitary transformation stored separately.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.pes_mn[-1,0].transform_ss

    traj.pes_mn[-1,0].ham_diag_ss
    """
    
    eval, evec = np.linalg.eigh(traj.pes_mn[-1,0].ham_diab_ss)
    traj.pes_mn[-1,0].transform_ss = evec
    # eval returned as column vector by np
    traj.pes_mn[-1,0].ham_diag_ss = np.diag(eval)

def adjust_energy(traj: Trajectory):
    """
    Adjusts potential energy to be consistent with throughout the simulation.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.par.ref_en
        first step only
    traj.pes_mn[-1,0].ham_diag_ss
    """
    
    # set ground-state energy as 0 at first step
    if traj.est.first: traj.par.ref_en = traj.pes_mn[-1,0].ham_diab_ss[0,0]
    # adjust potential energy to be consistent
    for s in range(traj.par.n_states): traj.pes_mn[-1,0].ham_diab_ss[s,s] -= traj.par.ref_en

def update_potential_energy(traj: Trajectory):
    """
    Updates potential energy based on the trajectory type.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.pes_mn[-1,0].poten
    """
    
    # simply active state PE for SH
    if traj.par.type == "sh": traj.pes_mn[-1,0].poten = traj.pes_mn[-1,0].ham_diag_ss[traj.hop.active, traj.hop.active]
    # population weighted PESs for MFE 
    elif traj.par.type == "mfe":
        traj.pes_mn[-1,0].poten = 0.
        for s in range(traj.par.n_states):
            traj.pes_mn[-1,0].poten += np.abs(traj.est.coeff_mns[-1,0,s])**2 * traj.pes_mn[-1,0].ham_diag_ss

def adjust_nacmes(traj: Trajectory):
    """
    Flips latest nacme based on overlap with the previous one.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.pes_mn[-1,0].nac_ddr_ssad

    traj.pes_mn[-1,0].nac_flip_ss
    """
    #Should check to ensure that this works with actual wf/ci overlaps
    

    for s1 in range(traj.par.n_states):
        for s2 in range(s1+1, traj.par.n_states):
            # calculate overlap
            if np.sum(traj.pes_mn[-2,0].nac_ddr_ssad[s1,s2] * traj.pes_mn[-1,0].nac_ddr_ssad[s1,s2]) < 0:
                # flip sign if overlap < 0 and set the flag
                traj.pes_mn[-1,0].nac_ddr_ssad[s1,s2] = -traj.pes_mn[-1,0].nac_ddr_ssad[s1,s2]
                traj.pes_mn[-1,0].nac_flip_ss[s1,s2] = True
                traj.pes_mn[-1,0].nac_flip_ss[s2,s1] = True
            else:
                traj.pes_mn[-1,0].nac_flip_ss[s1,s2] = False
                traj.pes_mn[-1,0].nac_flip_ss[s2,s1] = False
            # nacmes antisymmetric
            traj.pes_mn[-1,0].nac_ddr_ssad[s2,s1,:,:] = -traj.pes_mn[-1,0].nac_ddr_ssad[s1, s2, :, :]

def write_xyz(traj: Trajectory):
    """
    Creates an xyz file corresponding to the current geometry in AA.
    File is saved in the est subdirectory
    
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

    # eg molpro.xyz
    with open(f"{traj.est.file}.xyz", "w") as file:
        file.write(f"{traj.par.n_atoms}\n\n")
        for a in range(traj.par.n_atoms):
            file.write(f"{traj.geo_mn[-1,0].name_a[a]} \
                         {traj.geo_mn[-1,0].position_ad[a,0]*Constants.bohr2A} \
                         {traj.geo_mn[-1,0].position_ad[a,1]*Constants.bohr2A} \
                         {traj.geo_mn[-1,0].position_ad[a,2]*Constants.bohr2A}\n")

def run_model(traj: Trajectory):
    """
    Main routine for running model potentials.
    
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
    
    traj.est.run(traj)
    diagonalise_hamiltonian(traj)

def run_molpro(traj: Trajectory):
    """
    Runs molpro 202X calculation and reads the results.
    Creates input files and overwrites any previous outputs.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.pes_mn[-1,0].ham_diab_ss
        can include off-diagonal entries
    traj.pes_mn[-1,0].ham_diag_ss
        only diagonal entries
    traj.pes_mn[-1,0].transform_ss
        transforms between diabatic and diagonal hamiltonian
    traj.pes_mn[-1,0].nac_ddr_ssad
        only if requested
    traj.pes_mn[-1,0].overlap_ss
        only if nacmes not calculated
    traj.pes_mn[-1,0].dip_mom_ssd
        only if requested
    """

    # setup and calculations carried out in the est subdirectory
    os.chdir("est")

    # TODO: These are all temporary fixes, need to be better here
    if traj.ctrl.tdc_updater != 'nacme':
        traj.est.calc_nacs *= np.eye(traj.par.n_states)

    if traj.par.type == 'mfe' or len(traj.par.states) > 1:
        traj.est.calc_nacs[:,:] = 1

    write_xyz(traj)
    # creates an input file according to the specifics of traj.est
    create_input_molpro(traj.par.states, traj.est.file, traj.est.config, traj.est.calc_nacs, traj.est.skip, False)

    # perform molpro calculation
    err = os.system(f"molpro -W . -I . -d ./tmp_molpro -s {traj.est.file}.inp")
    # handle any errors in the process
    if err > 0:
        print(f'Error in MOLPRO, exit code {err}')
        raise EstCalculationBrokenError

    # indexing offset for different spin multiplicities
    ssum = np.cumsum(traj.par.states) - traj.par.states
    # reset hamiltonians and unitary transformation
    traj.pes_mn[-1,0].ham_diab_ss[:] = 0
    traj.pes_mn[-1,0].ham_diag_ss[:] = 0
    traj.pes_mn[-1,0].nac_ddr_ssad[:] = 0
    # assign hamiltonian entries yielded by molpro reader
    for s1, s2, i, j, val in read_output_molpro_ham(traj.est.file):
        # not all energy levels may be needed
        if i < traj.par.n_states and j < traj.par.n_states:
            traj.pes_mn[-1,0].ham_diab_ss[i+ssum[s1],j+ssum[s2]] += val
    # assign nacme and gradient entries yielded by molpro
    for s, i, j, a, val in read_output_molpro_nac(traj.est.file): traj.pes_mn[-1,0].nac_ddr_ssad[i+ssum[s],j+ssum[s],a] += val

    adjust_energy(traj)
    if traj.ctrl.diagonalise:
        # SH done in diagonal representation, need to transform
        diagonalise_hamiltonian(traj)

        # TODO: soc transformations, move later
        # need to transform gradient for non-diagonal hamiltonian
        # for details, see https://doi.org/10.1002/qua.2489
        g_diab = np.zeros((traj.par.n_states, traj.par.n_states, traj.par.n_atoms, 3), dtype=np.complex128)
        for i in range(traj.par.n_states):
            for j in range(traj.par.n_states):
                # on-diagonal part
                g_diab[i,j] = (i == j)*traj.pes_mn[-1,0].nac_ddr_ssad[i,i]
                # off-diagonal part
                g_diab[i,j] -= (traj.pes_mn[-1,0].ham_diab_ss[i,i] - traj.pes_mn[-1,0].ham_diab_ss[j,j])*traj.pes_mn[-1,0].nac_ddr_ssad[i,j]
        # just a big matrix multiplication with some extra dimensions
        g_diag = np.einsum("ij,jkad,kl->ilad", traj.pes_mn[-1,0].transform_ss.conj().T, g_diab, traj.pes_mn[-1,0].transform_ss)

        # only keep the real part of the gradient
        for i in range(traj.par.n_states):
            traj.pes_mn[-1,0].nac_ddr_ssad[i,i] = np.real(g_diag[i,i])

    else:
        # already diagonal
        traj.pes_mn[-1,0].ham_diag_ss[:] = traj.pes_mn[-1,0].ham_diab_ss
        traj.pes_mn[-1,0].transform_ss = np.eye(traj.par.n_states)

    # TODO: move this elsewhere
    overlap = traj.par.n_steps > 1
    overlap = traj.par.type != 'mfe'
    overlap = False
    # if nacmes were not calculated, calculate the wf overlap
    if overlap and traj.ctrl.substep == 0 and not traj.est.first:
        traj.pes_mn[-1,0].overlap_ss = run_wfoverlap_molpro(
            traj.est.file, traj.geo_mn[-1,0].name_a, traj.geo_mn[-1,0].position_ad, traj.geo_mn[-2,0].position_ad, 
            traj.est.config['basis'] , traj.par.n_states)
    
    # check for sign flips in nacmes
    if traj.par.n_steps > 1:
        adjust_nacmes(traj)

    # read dipole moments
    #traj.pes_mn[-1,0].dip_mom_ssd = read_output_molpro_dip(traj.est.file, traj.par.n_states)

    # return to main directory
    os.chdir("..")

def run_turbo(traj: Trajectory):
    """
    Runs ricc2 and riadc2 calculations from turbomole
    Uses turbomole_est library
    Uses wfoverlap from SHARC/Felix Plasser
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.pes_mn[-1,0].ham_diab_ss
        only diagonal entries
    traj.pes_mn[-1,0].ham_diag_ss
        only diagonal entries
    traj.pes_mn[-1,0].transform_ss
        only identity matrix (no SOC) 
    traj.pes_mn[-1,0].overlap_ss
    
    """
    
    os.chdir('est')
    run_ricc2(traj, traj.est.config)
    os.chdir('..')
    if traj.ctrl.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes_mn[-1,0].ham_diag_ss = traj.pes_mn[-1,0].ham_diab_ss
        traj.pes_mn[-1,0].transform_ss = np.identity(traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)

def run_pyscf_wrapper(traj: Trajectory):
    """
    Wrapper for the running of PySCF calculations.
    Links to pyscf_est.py module
    Runs mcscf and cisd calculations (don't use CISD, it's super slow)
    Currently only works with NACMEs, but intend to also do overlap soon.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    
    """
    
    os.chdir('est')
    if traj.est.config['type'] == 'cisd':
        run_pyscf_cisd(traj, traj.est.config)
    else:
        run_pyscf_mcscf(traj, traj.est.config)
    os.chdir('..')
    if traj.ctrl.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes_mn[-1,0].ham_diag_ss = traj.pes_mn[-1,0].ham_diab_ss
        traj.pes_mn[-1,0].transform_ss = np.identity(traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)

# write RC file
def run_molcas(traj: Trajectory):
    """
    Runs molcas electronic structure
    Uses molcas_est.py module
    Currently can perform CASSCF and (X/R)MS-CASPT2
    NACMEs implemented for both
    Overlaps are implemented at CASSCF level, i.e. only using the reference space for CASPT2
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.pes_mn[-1,0].ham_diab_ss
        only diagonal entries
    traj.pes_mn[-1,0].ham_diag_ss
        only diagonal entries
    traj.pes_mn[-1,0].transform_ss
        no SOC currently
    traj.pes_mn[-1,0].nac_ddr_ssad
        only if requested
    traj.pes_mn[-1,0].overlap_ss
        only if nacmes not calculated
    
    """
    
    traj.est.file = f"{traj.est.program}"

    os.chdir("est")
    skip = traj.est.skip


    if traj.ctrl.tdc_updater  != 'nacme':
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
            traj.pes_mn[-1,0].ham_diab_ss[i-skip,j-skip] = val

    traj.pes_mn[-1,0].overlap_ss = read_output_molcas_prop('molcas.log', traj.est.config)

    for s1 in range(traj.par.n_states):
        for s2 in range(s1+1):
            if traj.est.calculate_nacs[s1, s2]:
                if s1 == s2:
                    for i, i, a, val in read_output_molcas_grad(f"molcas.log", traj.est.config): traj.pes_mn[-1,0].nac_ddr_ssad[i-skip, i-skip, a] = val
                else:
                    for i, j, a, val in read_output_molcas_nac(f"molcas_{s2}_{s1}.log", s2, s1): traj.pes_mn[-1,0].nac_ddr_ssad[i-skip,j-skip,a] = val

    os.chdir("..")


    if traj.ctrl.diagonalise:
        diagonalise_hamiltonian(traj)
    else:
        traj.pes_mn[-1,0].ham_diag_ss = traj.pes_mn[-1,0].ham_diab_ss
        traj.pes_mn[-1,0].transform_ss = np.identity(traj.par.n_states)

    adjust_energy(traj)
    adjust_nacmes(traj)
