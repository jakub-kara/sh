import numpy as np

from classes import Trajectory

def get_com_position(traj: Trajectory) -> np.ndarray:
    """
    Calculate the centre-of-mass position of a molecule.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    com: np.ndarray
        centre of mass
    
    Modifies
    --------
    None
    """
    
    total_mass = np.sum(traj.geo_mn[-1,0].mass_a)
    com = np.sum(traj.geo_mn[-1,0].position_ad * traj.geo_mn[-1,0].mass_a[:,None], axis=0)/total_mass
    return com

def set_com_position(traj: Trajectory) -> None:
    """
    Sets the centre-of-mass position of a molecule to the origin.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.geo_mn[-1,0].position_ad
    """
    
    traj.geo_mn[-1,0].position_ad -= get_com_position(traj)

def get_com_velocity(traj: Trajectory) -> np.ndarray:
    """
    Calculate the centre-of-mass velocity of a molecule.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    com_vel: np.ndarray
        centre-of-mass velocity
    
    Modifies
    --------
    None
    """
    
    total_mass = np.sum(traj.geo_mn[-1,0].mass_a)
    com_vel = np.sum(traj.geo_mn[-1,0].velocity_ad * traj.geo_mn[-1,0].mass_a[:,None], axis=0)/total_mass
    return com_vel

def set_com_velocity(traj: Trajectory) -> None:
    """
    Sets the centre-of-mass velocity of a molecule to zero.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.geo_mn[-1,0].velocity_ad
    """
    
    traj.geo_mn[-1,0].velocity_ad -= get_com_velocity(traj)

def set_rotation(traj: Trajectory) -> None:
    """
    Removes the rotation of a molecule.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.geo_mn[-1,0].velocity_ad
    """

    # inertia tensor
    # defined as I = sum_a m_a * (|x_a|**2 * E_3 - x_a @ x_a.T)
    inertia = np.einsum(
        "a,aij->ij", traj.geo_mn[-1,0].mass_a, 
        np.einsum("ij,al->aij", np.eye(3), traj.geo_mn[-1,0].position_ad**2) - 
        np.einsum("ai,aj->aij", traj.geo_mn[-1,0].position_ad, traj.geo_mn[-1,0].position_ad))
    # momentum of each atom
    mom = np.sum(traj.geo_mn[-1,0].velocity_ad * traj.geo_mn[-1,0].mass_a, axis=0)
    # angular momentum of each atom assuming com is origin
    ang_mom = np.cross(traj.geo_mn[-1,0].position_ad, mom)
    # angular velocity of each atom
    ang_vel = np.linalg.inv(inertia) @ ang_mom
    # linear velocity of each atom
    vel = np.cross(ang_vel, traj.geo_mn[-1,0].position_ad)
    # subtract linear velocities
    traj.geo_mn[-1,0].velocity_ad -= vel

def get_kinetic_energy(traj: Trajectory):
    """
    Calculates kinetic energy of a molecule.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    float
        kinetic energy
    
    Modifies
    --------
    None
    """
    
    return 0.5*np.einsum("a,ai->", traj.geo_mn[-1,0].mass_a, traj.geo_mn[-1,0].velocity_ad**2)

def get_total_momentum(traj: Trajectory) -> np.ndarray:
    """
    Calculates total momentum of a molecule.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    np.ndarray
        total momentum
    
    Modifies
    --------
    None
    """
    
    return np.einsum("a,ai->i", traj.geo_mn[-1,0].mass_a, traj.geo_mn[-1,0].velocity_ad)

def get_total_ang_momentum(traj: Trajectory) -> np.ndarray:
    """
    Calculates total angular momentum of a molecule wrt origin.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    total_ang_mom: np.ndarray
        total angular momentum
    
    Modifies
    --------
    
    """

    return np.einsum("a,ai->i", traj.geo_mn[-1,0].mass_a, np.cross(traj.geo_mn[-1,0].position_ad, traj.geo_mn[-1,0].velocity_ad))