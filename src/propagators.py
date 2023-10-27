import numpy as np
import scipy
import time
import pickle

from classes import TrajectorySH, SimulationSH
from hopping import get_hopping_probability_state_coeff, check_hop, check_hop_mash, adjust_velocity_and_hop, decoherence_zhu
from io_methods import write_to_xyz, back_up_step
from abinitio import get_electronic_structure
from kinematics import *

import fortran_modules.propagators_f as propagators_f


def velocity_verlet_position_sh(traj: TrajectorySH, ctrl: SimulationSH):
    traj.geo.position_ad += \
        traj.geo.velocity_ad*ctrl.dt + 0.5*traj.geo.force_sad[traj.hop.active]/traj.geo.mass_a[:,None]*ctrl.dt**2
    
def velocity_verlet_velocity_sh(traj: TrajectorySH, ctrl: SimulationSH):
    traj.geo.velocity_ad += \
        0.5*traj.geo.force_sad[traj.hop.active]/traj.geo.mass_a[:,None]*ctrl.dt

def update_tdc_nac_dr(traj: TrajectorySH):
    traj.est.pes.nac_ddt_ss = np.zeros((traj.est.n_states, traj.est.n_states))
    for s1 in range(traj.est.n_states):
        for s2 in range(traj.est.n_states):
            if s1 != s2:
                traj.est.pes.nac_ddt_ss[s1,s2] = np.sum(traj.geo.velocity_ad[:,:] * traj.est.pes.nac_ddr_ssad[s1,s2,:,:])/ \
                    (traj.est.pes.ham_diag_ss[s2] - traj.est.pes.ham_diag_ss[s1])

def get_state_coeff_values(traj: TrajectorySH, ctrl: SimulationSH, shift: float, prev: np.ndarray):
    temp_coeff_s = traj.hop.state_coeff_s + shift*prev*ctrl.dtq
    frac = (ctrl.qstep+shift)/ctrl.quantum_resolution
    energy_s = frac*traj.est.pes.ham_diag_ss + (1-frac)*traj.est.pes_old.ham_diag_ss
    nac_dt_ss = frac*traj.est.pes.nac_ddt_ss + (1-frac)*traj.est.pes.nac_ddt_ss
    coeff_s = np.zeros(traj.est.n_states, dtype=complex)
    for s in range(traj.est.n_states):
        coeff_s[s] = -1.j*energy_s[s,s]*temp_coeff_s[s] - np.inner(nac_dt_ss[s,:], temp_coeff_s[:])
    return coeff_s

def state_coeff_rk4(traj: TrajectorySH, ctrl: SimulationSH):
    k1 = get_state_coeff_values(traj, ctrl, 0, 0)
    k2 = get_state_coeff_values(traj, ctrl, 0.5, k1)
    k3 = get_state_coeff_values(traj, ctrl, 0.5, k2)
    k4 = get_state_coeff_values(traj, ctrl, 1, k3)
    traj.hop.state_coeff_s += ctrl.dtq/6*(k1+2*k2+2*k3+k4)

def propagate_state_coeff(traj: TrajectorySH, ctrl: SimulationSH):
    for ctrl.qstep in range(ctrl.quantum_resolution):
        #propagators_f.state_coeff_rk4(traj.hop.state_coeff_s, traj.est.pes.nac_dt_ss, traj.est.pes.energy_s, ctrl.dtq)
        state_coeff_rk4(traj, ctrl)
        if traj.hop.target == traj.hop.active:
            get_hopping_probability_state_coeff(traj, ctrl.dtq)
            check_hop(traj)
    #propagators_f.normalise_state_coeff(traj.hop.state_coeff_s)

def initialise_dynamics(traj: TrajectorySH, ctrl: SimulationSH):
    traj.est.pes_old = traj.est.pes
    get_electronic_structure(traj, ctrl)
    traj.est.first = False

    back_up_step(traj, ctrl)
    write_to_xyz(traj, ctrl)

def loop_dynamics(traj: TrajectorySH, ctrl: SimulationSH):
    while not (lambda ctrl, traj: eval(ctrl.termination_cond))(ctrl, traj):
        print(traj.est.pes.ham_diag_ss)
        print(traj.est.pes.nac_ddr_ssad)

        t1 = time.time()
        # step 1: position update & 1/2 velocity update
        propagators_f.update_position_velocity_verlet(
            traj.geo.position_ad, traj.geo.velocity_ad, traj.geo.force_sad[traj.hop.active], traj.geo.mass_a, ctrl.dt
        )
        propagators_f.update_velocity_velocity_verlet(
            traj.geo.velocity_ad, traj.geo.force_sad[traj.hop.active], traj.geo.mass_a, ctrl.dt
        )

        t2 = time.time()
        # step 2: electronic structure
        traj.est.pes_old = traj.est.pes
        get_electronic_structure(traj, ctrl)

        t3 = time.time()
        # step 3: 2/2 velocity update
        propagators_f.update_velocity_velocity_verlet(
            traj.geo.velocity_ad, traj.geo.force_sad[traj.hop.active], traj.geo.mass_a, ctrl.dt
        )

        t4 = time.time()
        # step 4&5: update electronic wf coefficients & compute hopping probabilities
        traj.est.pes.nac_ddt_ss = propagators_f.update_tdc(traj.geo.velocity_ad, traj.est.pes.nac_ddr_ssad)
        propagate_state_coeff(traj, ctrl)

        t5 = time.time()
        # step 6: select a new active state
        adjust_velocity_and_hop(traj)
        decoherence_zhu(traj, ctrl.dt)

        t6 = time.time()

        # step 7: recalculate EST if hop occured
        if traj.est.recalculate:
            get_electronic_structure(traj, ctrl)
            traj.est.recalculate = False

        # increment time
        ctrl.current_time += ctrl.dt
        ctrl.step += 1
        back_up_step(traj, ctrl)
        write_to_xyz(traj, ctrl)

        if 0:
            print("nuclear 1/2: ", t2-t1)
            print("est: ", t3-t2)
            print("nuclear 2/2: ", t4-t3)
            print("coeff: ", t5-t4)
            print("active: ", t6-t5)
            print("total: ", t6-t1, "\n")