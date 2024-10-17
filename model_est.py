from classses.classes import Trajectory
import fmodules.models_f as models_f
from abinitio import diagonalise_hamiltonian, adjust_nacmes

def harm(traj: Trajectory):
    """
    1D Harmonic potenial
    
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

    traj.pes_mn[-1,0].ham_diag_ss

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss[0,0] = 0.5*traj.geo_mn[-1,0].position_ad[0,0]**2
    traj.pes_mn[-1,0].ham_diag_ss = traj.pes_mn[-1,0].ham_diab_ss
    traj.pes_mn[-1,0].nac_ddr_ssad[0,0,0,0] = traj.geo_mn[-1,0].position_ad[0,0]

def spin_boson(traj: Trajectory):
    """
    1D spin-boson hamiltonian
    Details
    
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

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, gradH_ssad = models_f.spin_boson(traj.geo_mn[-1,0].position_ad)
    diagonalise_hamiltonian(traj)
    traj.pes_mn[-1,0].nac_ddr_ssad = models_f.get_nac_and_gradient(traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, gradH_ssad)
    #adjust_nacmes(traj)

def tully_1(traj: Trajectory):
    """
    Original 1st Tully models
    
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

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, gradH_ssad = models_f.tully_1(traj.geo_mn[-1,0].position_ad)
    diagonalise_hamiltonian(traj)
    traj.pes_mn[-1,0].nac_ddr_ssad = models_f.get_nac_and_gradient(traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, gradH_ssad)
    #adjust_nacmes(traj)

def tully_s(traj: Trajectory):
    """
    Modified 1st Tully model with continuous derivatives
    
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

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, gradH_ssad = models_f.tully_s(traj.geo_mn[-1,0].position_ad)
    diagonalise_hamiltonian(traj)
    traj.pes_mn[-1,0].nac_ddr_ssad = models_f.get_nac_and_gradient(traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, gradH_ssad)
    adjust_nacmes(traj)

def tully_n(traj: Trajectory):
    """
    Several repeated tully_s avoided crossings
    
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

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, gradH_ssad = models_f.tully_n(traj.geo_mn[-1,0].position_ad)
    diagonalise_hamiltonian(traj)
    traj.pes_mn[-1,0].nac_ddr_ssad = models_f.get_nac_and_gradient(traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, gradH_ssad)
    adjust_nacmes(traj)

def tully_2(traj: Trajectory):
    """
    2nd Tully model
    
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

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, gradH_ssad = models_f.tully_2(traj.geo_mn[-1,0].position_ad)
    diagonalise_hamiltonian(traj)
    traj.pes_mn[-1,0].nac_ddr_ssad = models_f.get_nac_and_gradient(traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, gradH_ssad)
    adjust_nacmes(traj)

def tully_3(traj: Trajectory):
    """
    3rd Tully model
    
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

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, gradH_ssad = models_f.tully_3(traj.geo_mn[-1,0].position_ad)
    diagonalise_hamiltonian(traj)
    traj.pes_mn[-1,0].nac_ddr_ssad = models_f.get_nac_and_gradient(traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, gradH_ssad)
    adjust_nacmes(traj)

def sub_x(traj: Trajectory):
    """
    Subotnik "X" model
    
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

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, gradH_ssad = models_f.sub_x(traj.geo_mn[-1,0].position_ad)
    diagonalise_hamiltonian(traj)
    traj.pes_mn[-1,0].nac_ddr_ssad = models_f.get_nac_and_gradient(traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, gradH_ssad)
    adjust_nacmes(traj)

def sub_s(traj: Trajectory):
    """
    Subotnik "S" model
    
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

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, gradH_ssad = models_f.sub_s(traj.geo_mn[-1,0].position_ad)
    diagonalise_hamiltonian(traj)
    traj.pes_mn[-1,0].nac_ddr_ssad = models_f.get_nac_and_gradient(traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, gradH_ssad)
    adjust_nacmes(traj)

def sub_2(traj: Trajectory):
    """
    Subotnik 2D model
    
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

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, gradH_ssad = models_f.sub_2(traj.geo_mn[-1,0].position_ad)
    diagonalise_hamiltonian(traj)
    traj.pes_mn[-1,0].nac_ddr_ssad = models_f.get_nac_and_gradient(traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, gradH_ssad)

def lvc_wrapper(traj: Trajectory):
    """
    UNREALIABLE, DO NOT USE
    LVC model
    
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

    traj.pes_mn[-1,0].ham_diag_ss

    traj.pes_mn[-1,0].transform_ss

    traj.pes_mn[-1,0].nac_ddr_ssad
    """
    
    traj.pes_mn[-1,0].ham_diab_ss, traj.pes_mn[-1,0].ham_diag_ss, traj.pes_mn[-1,0].transform_ss, traj.pes_mn[-1,0].nac_ddr_ssad = \
        LVC.get_est(traj.geo_mn[-1,0].position_ad.flatten())