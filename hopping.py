import numpy as np

from classes import Trajectory
from kinematics import get_kinetic_energy
from errors import *
from io_methods import write_log

def get_hopping_prob_ddt(traj: Trajectory):
    """
    Calculates hopping probability based on tdc.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.hop.prob_s
    """
    
    for s in range(traj.par.n_states):
        # assign 0 hopping probability to active state
        if s == traj.hop.active:
            traj.hop.prob_s[s] = 0
        # standard Tully-based hopping probability
        else:
            # TODO: check timestep changes with variable timestep
            prob = np.real(traj.pes_mn[-1,0].nac_ddt_ss[s, traj.hop.active] * np.conj(traj.est.coeff_mns[-1,0,traj.hop.active]) * traj.est.coeff_mns[-1,0,s])
            prob *= -2 * traj.ctrl.dtq / np.abs(traj.est.coeff_mns[-1,0,traj.hop.active])**2 
            traj.hop.prob_s[s] = max(0, prob)


def get_hopping_prob_LD(traj: Trajectory, R):
    """
    calculated the hopping probability using SHARCs LD formula
    should only be called in the Local diabatisation section of the code
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.hop.prob_s
    
    """
    
    for s in range(traj.par.n_states):
        if s == traj.hop.active:
            traj.hop.prob_s[s] = 0
        else:
            prob = (1 - np.abs(traj.est.coeff_mns[-1,0,traj.hop.active])**2/np.abs(traj.est.coeff_mns[-2,0,traj.hop.active])**2) 
            prob *= np.real(traj.est.coeff_mns[-1,0,s]*np.conj(R[s,traj.hop.target])*np.conj(traj.est.coeff_mns[-2,0,traj.hop.active]))
            prob /= (np.abs(traj.est.coeff_mns[-2,0,traj.hop.active])**2-np.real(traj.est.coeff_mns[-1,0,traj.hop.active]*np.conj(R[traj.hop.active, traj.hop.active])*np.conj(traj.est.coeff_mns[-2,0,traj.hop.active])))
            traj.hop.prob_s[s] = max(0, prob)

def check_hop(traj: Trajectory):
    """
    Checks whether a hop was succesful based on the type of SH.
    
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
    
    if traj.hop.type == 'mash':
        check_hop_mash(traj)
    elif traj.hop.type == 'fssh':
        check_hop_fssh(traj)
    else:
        raise HoppingTypeNotFoundError

def check_hop_fssh(traj: Trajectory):
    """
    Determines a hop target based on a random number and the hopping probabilities.
    The target needs to get checked for conservation of energy.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.hop.target
    """
    
    r = np.random.uniform()
    cum_prob = 0
    for s in range(traj.par.n_states):
        cum_prob += traj.hop.prob_s[s]
        if r < cum_prob:
            traj.hop.target = s
            return

def check_hop_mash(traj: Trajectory):
    """
    Determines a hop target based on the maximum quantum population.
    The target needs to get checked for conservation of energy.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.hop.target
    """
    
    if traj.ctrl.qstep + 1 == traj.par.n_qsteps:  # to be renamed!
        traj.hop.target = np.argmax(np.abs(traj.est.coeff_mns[-1,0]))


def adjust_velocity_and_hop(traj: Trajectory):
    """
    Assesses if there is enough energy to perform a hop.
    Then adjusts velocity to conserve energy.
    Reworked version from https://doi.org/10.1016/j.chemphys.2008.01.044
        In general reworked to take in arbitrary directions
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.hop.active

    traj.hop.target

    traj.geo_mn[-1,0].velocity_ad
        only if hop accepted or reverse flag set
    traj.ctrl.init_steps
        only if hop accepted
    """
    
    # normalises vector
    def normalise(a):
        return a / np.linalg.norm(a)

    # projects vector a onto vector d
    def project(a, d):
        return normalise(d) * np.sum(a * normalise(d))
    
    # if target not reassigned, return
    if traj.hop.active == traj.hop.target:
        return
        
    write_log(traj, f"Attempting to hop from {traj.hop.active} to {traj.hop.target}\n")

    # set delta as the direction of rescaling
    if traj.hop.rescale == 'ddr':
        # rescale along ddr
        delta = normalise(traj.pes_mn[-1,0].nac_ddr_ssad[traj.hop.active,traj.hop.target])
        delta /= traj.geo_mn[-1,0].mass_a[:,None]
    elif traj.hop.rescale == 'mash':
        # rescale along expression E3 in https://doi.org/10.1063/5.0158147
        delta = np.zeros_like(traj.geo_mn[-1,0].velocity_ad)
        for i in range(traj.par.n_states):
            delta += np.real(np.conj(traj.est.coeff_mns[-1,0,i]) * traj.pes_mn[-1,0].nac_ddr_ssad[i, traj.hop.active] * traj.est.coeff_mns[-1,0,traj.hop.active] - 
                             np.conj(traj.est.coeff_mns[-1,0,i]) * traj.pes_mn[-1,0].nac_ddr_ssad[i, traj.hop.target] * traj.est.coeff_mns[-1,0,traj.hop.target])
        delta /= traj.geo_mn[-1,0].mass_a[:,None]
        delta = normalise(delta)
    else:
        # rescale uniformly
        delta = normalise(traj.geo_mn[-1,0].velocity_ad)

    ediff = traj.pes_mn[-1,0].ham_diag_ss[traj.hop.active,traj.hop.active] - traj.pes_mn[-1,0].ham_diag_ss[traj.hop.target,traj.hop.target]

    # compute coefficients in the quadratic equation
    a = 0.5 * np.sum(traj.geo_mn[-1,0].mass_a[:, None] * delta * delta)
    b = -np.sum(traj.geo_mn[-1,0].mass_a[:, None] * traj.geo_mn[-1,0].velocity_ad * delta)
    c = -ediff

    # TODO: move to settings/control
    reverse = False

    # find the determinant
    D = b**2 - 4 * a * c
    if D < 0:
        # reverse if no real solution and flag set
        if reverse:
            gamma = -b/a
        else:
            gamma = 0
    # choose the smaller solution of the two
    elif b < 0:
        gamma = -(b + np.sqrt(D)) / (2 * a)
    elif b >= 0:
        gamma = -(b - np.sqrt(D)) / (2 * a)
    
    # adjust velocity
    traj.geo_mn[-1,0].velocity_ad -= gamma * delta

    # hop accepted, set quantities to correspond to the new active state
    if D > 0:
        traj.hop.active = traj.hop.target
        # recalculate the gradient
        traj.est.nacs_setter(traj, False)
        # soft restart for multistep integrators
        traj.ctrl.init_steps = traj.ctrl.nuc_loop_scheme.m

        write_log(traj, "Hop succeeded\n")
    else:
        # reset hop target
        traj.hop.target = traj.hop.active

        write_log(traj, "Hop failed")
        if reverse: write_log(traj, ", velocity reversed along NACME")
        write_log(traj, "\n")

def decoherence_edc(traj: Trajectory, c=0.1):
    """
    Performs energy-based decoherence correction (EDC) on WF coefficients.
    
    Parameters
    ----------
    traj: Trajectory
        Trajectory object
    c: float
        Empirical exponential decay coefficient. Default is 0.1 Eh.
    
    Returns
    -------
    None
    
    Modifies
    --------
    traj.est.coeff_mns
    """
    
    # get total kinetic energy
    kinetic_energy = get_kinetic_energy(traj)
    for s in range(traj.par.n_states):
        # do not decohere the active state
        if s == traj.hop.active:
            continue
        else:
            # get decay rate based on energy gap between states
            decay_rate = 1/np.abs(traj.pes_mn[-1,0].ham_diag_ss[s,s] - traj.pes_mn[-1,0].ham_diag_ss[traj.hop.active, traj.hop.active])*(1 + c/kinetic_energy)
            # reduce the WF coefficient by the decay factor
            traj.est.coeff_mns[-1,0,s] *= np.exp(-traj.ctrl.dt/decay_rate)

    # renormalise quantum populations
    tot_pop = np.sum(np.abs(traj.est.coeff_mns[-1,0])**2) - np.abs(traj.est.coeff_mns[-1,0,traj.hop.active])**2
    traj.est.coeff_mns[-1,0,traj.hop.active] *= np.sqrt(1 - tot_pop)/np.abs(traj.est.coeff_mns[-1,0,traj.hop.active])
