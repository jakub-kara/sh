import numpy as np

from classes import TrajectorySH, SimulationSH
from kinematics import get_kinetic_energy

def get_hopping_probability_state_coeff(traj: TrajectorySH, dtq: float):
    for s in range(traj.est.n_states):
        if s == traj.hop.active:
            traj.hop.prob_s[s] = 0
        else:
            prob = np.real(traj.est.pes.nac_ddt_ss[s,traj.hop.active] * \
                np.conj(traj.hop.state_coeff_s[traj.hop.active])*traj.hop.state_coeff_s[s])
            prob *= -2*dtq/np.abs(traj.hop.state_coeff_s[traj.hop.active])**2
            traj.hop.prob_s[s] = max(0, prob)

def check_hop(traj: TrajectorySH):
    r = np.random.uniform()
    cum_prob = 0
    for s in range(traj.est.n_states):
        cum_prob += traj.hop.prob_s[s]
        if r < cum_prob:
            traj.hop.target = s
            return

def check_hop_mash(traj: TrajectorySH):
    traj.hop.target = np.argmax(np.abs(traj.hop.state_coeff_s))

def adjust_velocity_and_hop(traj: TrajectorySH):
    #https://doi.org/10.1016/j.chemphys.2008.01.044
    if traj.hop.active == traj.hop.target:
        return
    
    A = 0
    B = traj.est.pes.nac_ddt_ss[traj.hop.active, traj.hop.target]
    for a in range(traj.geo.n_atoms):
        d = traj.est.pes.nac_ddr_ssad[traj.hop.active, traj.hop.target, a, :]
        A += np.inner(d, d)/traj.geo.mass_a[a]
    A /= 2

    D = B**2 + 4*A*(traj.est.pes.ham_diag_ss[traj.hop.active, traj.hop.active] - traj.est.pes.ham_diag_ss[traj.hop.target, traj.hop.target])
    if D < 0:
        G = B/A
    elif B < 0:
        G = (B + np.sqrt(D))/2/A
    else:
        G = (B - np.sqrt(D))/2/A
    for a in range(traj.geo.n_atoms):
        traj.geo.velocity_ad[a,:] -= G*np.real(traj.est.pes.nac_ddr_ssad[traj.hop.active, traj.hop.target, a, :])/traj.geo.mass_a[a]
    
    if D > 0:
        traj.hop.active = traj.hop.target
        traj.est.recalculate = True

def decoherence_zhu(traj: TrajectorySH, dt: float, c=0.1):
    kinetic_energy = get_kinetic_energy(traj)
    amp_sum = 0.
    for s in range(traj.est.n_states):
        if s == traj.hop.active:
            continue
        else:
            decay_rate = 1/np.abs(traj.est.pes.ham_diag_ss[s,s] - traj.est.pes.ham_diag_ss[traj.hop.active, traj.hop.active])*(1 + c/kinetic_energy)
            traj.hop.state_coeff_s[s] *= np.exp(-dt/decay_rate)
            amp_sum += np.abs(traj.hop.state_coeff_s[s])**2

    traj.hop.state_coeff_s[traj.hop.active] *= np.sqrt(1 - amp_sum)/ \
        np.abs(traj.hop.state_coeff_s[traj.hop.active])

"""
def propagate_moments(traj: TrajectorySH, dt: float):
    lam = traj.hop.active
    del_force = np.zeros((traj.n_states, traj.n_atoms, 3))
    acc_decoh = np.zeros((traj.n_states, traj.n_atoms, 3))

    np.copyto(del_force, traj.force_sad)
    temp = del_force[lam,:,:]
    for s in range(traj.n_states):
        del_force[s,:,:] -= temp
        for d in range(3):
            acc_decoh[s,:,d] = del_force[s,:,d]*np.abs(traj.state_coeff_s[s])**2/traj.mass_a

    for s in range(traj.n_states):
        for d in range(3):
            traj.del_position[s,:,d] += traj.del_momentum[s,:,d]/traj.mass_a*dt + 0.5*acc_decoh[s,:,d]*dt**2
            traj.del_momentum[s,:,d] += 0.5*traj.mass_a*acc_decoh[s,:,d]*dt

    del_force[:,:,:] = 0.
    for s1 in range(traj.n_states):
        for s2 in range(traj.n_states):
            del_force[s1,:,:] += np.abs(traj.overlap_ss[s1,s2]**2)* \
                (traj.force_sad[s2,:,:] - traj.force_sad[lam,:,:])
    
    for s in range(traj.n_states):
        for d in range(3):
            acc_decoh[s,:,d] = del_force[s,:,d]*np.abs(traj.state_coeff_s[s])**2/traj.mass_a
            traj.del_momentum[s,:,d] += 0.5*traj.mass_a*acc_decoh[s,:,d]*dt
        
    temp_del_pos = np.zeros((traj.n_states, traj.n_atoms, 3), dtype=np.complex128)
    temp_del_mom = np.zeros((traj.n_states, traj.n_atoms, 3), dtype=np.complex128)
    for s1 in range(traj.n_states):
        for s2 in range(traj.n_states):
            temp_del_pos[s1,:,:] += np.abs(traj.overlap_ss[s2,s1]**2)*traj.del_position[s2,:,:]
            temp_del_mom[s1,:,:] += np.abs(traj.overlap_ss[s2,s1]**2)*traj.del_momentum[s2,:,:]
    
    traj.del_position[:,:,:] = temp_del_pos[:,:,:]
    traj.del_momentum[:,:,:] = temp_del_mom[:,:,:]

def collapse_state_coeffs(traj: TrajectorySH, dt: float):
    lam = traj.active

    for s in range(traj.n_states):
        if s != traj.active:
            reset_prob = np.sum((traj.force_sad[s,:,:] - traj.force_sad[lam,:,:]) * \
                np.real(traj.del_position[s,:,:] - traj.del_position[lam,:,:]))/2

            temp = np.abs((traj.energy_s[lam] - traj.energy_s[s])*traj.nac_dt_ss[lam,s] * \
                np.sum((traj.del_position[s,:,:] - traj.del_position[lam,:,:]) * -traj.velocity_ad)) / \
                np.sum(traj.velocity_ad**2)
            collapse_prob = reset_prob - 2*temp
            collapse_prob *= dt
            reset_prob *= -dt

            r = np.random.uniform()
            if r < collapse_prob:
                traj.state_coeff_s[lam] *= (np.sqrt(np.abs(traj.state_coeff_s[lam])**2 + \
                    np.abs(traj.state_coeff_s[s])**2))/np.abs(traj.state_coeff_s[lam])
                traj.state_coeff_s[s] = 0

            if r < collapse_prob or r < reset_prob:
                traj.del_position[s,:] = 0
                traj.del_momentum[s,:] = 0
"""