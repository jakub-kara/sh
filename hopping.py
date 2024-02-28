import numpy as np

from classes import Trajectory
from kinematics import get_kinetic_energy
from errors import *
from io_methods import write_log

def get_hopping_prob_ddr(traj: Trajectory):
    for s in range(traj.par.n_states):
        if s == traj.hop.active:
            traj.hop.prob_s[s] = 0
        else:
            prob = np.real(traj.pes.nac_ddt_mnss[-1, 0, s, traj.hop.active] * np.conj(traj.est.coeff_mns[-1,0,traj.hop.active]) * traj.est.coeff_mns[-1,0,s])
            prob *= -2 * traj.ctrl.dtq / np.abs( traj.est.coeff_mns[-1,0,traj.hop.active])**2 # check timestep changes with variable timestep
            traj.hop.prob_s[s] = max(0, prob)

def check_hop(traj: Trajectory):
    if traj.hop.type == 'mash':
        check_hop_mash(traj)
    elif traj.hop.type == 'fssh':
        check_hop_fssh(traj)
    else:
        raise HoppingTypeNotFoundError


def check_hop_fssh(traj: Trajectory):
    r = np.random.uniform()
    cum_prob = 0
    for s in range(traj.par.n_states):
        cum_prob += traj.hop.prob_s[s]
        if r < cum_prob:
            traj.hop.target = s
            return

def check_hop_mash(traj: Trajectory):
    if traj.ctrl.qstep + 1 == traj.par.n_qsteps:  # to be renamed!
        traj.hop.target = np.argmax(np.abs(traj.est.coeff_mns[-1,0]))


def normalise(a):
    return a / np.linalg.norm(a)


def project(a, d):
    return normalise(d) * np.sum(a * normalise(d))


def adjust_velocity_and_hop(traj: Trajectory):
    # Reworked version from https://doi.org/10.1016/j.chemphys.2008.01.044

    if traj.hop.active == traj.hop.target:
        return
    print(f'Attempting to hop from {traj.hop.active} to {traj.hop.target}')
    write_log(traj, f"Attempting to hop from {traj.hop.active} to {traj.hop.target}\n")

    # set delta as the direction of rescaling
    if traj.hop.rescale == 'ddr':
        # rescale along ddr
        delta = normalise(traj.pes.nac_ddr_mnssad[-1,0,traj.hop.active,traj.hop.target])
        delta /= traj.geo.mass_a[:,None]
    elif traj.hop.rescale == 'mash':
        # rescale along expression E3 in https://doi.org/10.1063/5.0158147
        delta = np.zeros_like(traj.geo.velocity_mnad[-1,0])
        for i in range(traj.par.n_states):
            delta += np.real(np.conj(traj.est.coeff_mns[-1,0,i]) * traj.pes.nac_ddr_mnssad[-1,0,i, traj.hop.active] * traj.est.coeff_mns[-1,0,traj.hop.active] - 
                np.conj(traj.est.coeff_mns[-1,0,i]) * traj.pes.nac_ddr_mnssad[-1,0,i, traj.hop.target] * traj.est.coeff_mns[-1,0,traj.hop.target])
        delta /= traj.geo.mass_a[:,None]
        delta = normalise(delta)
    else:
        # rescale uniformly
        delta = normalise(traj.geo.velocity_mnad[-1,0,:,:])

    ediff = traj.pes.ham_diag_mnss[-1,0,traj.hop.active,traj.hop.active] - traj.pes.ham_diag_mnss[-1,0,traj.hop.target,traj.hop.target]

    a = 0.5 * np.sum(traj.geo.mass_a[:, None] * delta * delta)
    b = -np.sum(traj.geo.mass_a[:, None] * traj.geo.velocity_mnad[-1,0] * delta)
    c = -ediff
    reverse = True

    D = b**2 - 4 * a * c
    if D < 0:
        if reverse:
            gamma = -b/a
        else:
            gamma = 0
    elif b < 0:
        gamma = -(b + np.sqrt(D)) / (2 * a)
    elif b >= 0:
        gamma = -(b - np.sqrt(D)) / (2 * a)

    traj.geo.velocity_mnad[-1,0,:,:] -= gamma * delta

    if D > 0:
        traj.hop.active = traj.hop.target
        traj.est.nacs_setter(traj, False)
        traj.geo.force_mnad = -traj.pes.nac_ddr_mnssad[:,:,traj.hop.active,traj.hop.active,:,:]/traj.geo.mass_a[None,None,:,None]
        traj.ctrl.init_steps = traj.par.n_steps
        print("Hop succeeded")
        write_log(traj, "Hop succeeded\n")
    else:
        traj.hop.target = traj.hop.active
        print("Hop failed")
        write_log(traj, "Hop failed")
        if reverse: write_log(traj, ", velocity reversed along NACME")
        write_log(traj, "\n")

def decoherence_edc(traj: Trajectory, c=0.1):
    'energy-based decoherence correction'
    kinetic_energy = get_kinetic_energy(traj)
    amp_sum = 0.
    for s in range(traj.par.n_states):
        if s == traj.hop.active:
            continue
        else:
            decay_rate = 1/np.abs(traj.pes.ham_diag_mnss[-1, -1, s, s] - traj.pes.ham_diag_mnss[-1, -1, traj.hop.active, traj.hop.active])*(1 + c/kinetic_energy)
            traj.est.coeff_mns[-1,0,s] *= np.exp(-traj.ctrl.dt/decay_rate)
            amp_sum += np.abs(traj.est.coeff_mns[-1,0,s])**2

    traj.est.coeff_mns[-1,0,traj.hop.active] *= np.sqrt(1 - amp_sum)/np.abs(traj.est.coeff_mns[-1,0,traj.hop.active])
