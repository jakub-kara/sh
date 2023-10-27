import numpy as np

from classes import TrajectorySH, SimulationSH

def get_com_position(traj: TrajectorySH):
    total_mass = np.sum(traj.geo.mass_a)
    
    com = np.zeros(3)
    for a in range(traj.geo.n_atoms):
        com += traj.geo.position_ad[a,:]*traj.geo.mass_a[a]
    com /= total_mass
    return com

def set_com_position(traj: TrajectorySH):
    traj.geo.position_ad -= get_com_position(traj)

def get_com_velocity(traj: TrajectorySH):
    total_mass = np.sum(traj.geo.mass_a)

    com_vel = np.zeros(3)
    for a in range(traj.geo.n_atoms):
        com_vel += traj.geo.velocity_ad[a,:]*traj.geo.mass_a[a]
    com_vel /= total_mass
    return com_vel

def set_com_velocity(traj: TrajectorySH):
    traj.geo.velocity_ad -= get_com_velocity(traj)

def set_rotation(traj: TrajectorySH):
    inertia = np.zeros((3,3))

    for a in range(traj.geo.n_atoms):
        inertia[0,0] += traj.geo.mass_a[a]*(traj.geo.position_ad[a,1]**2 + traj.geo.position_ad[a,2]**2)
        inertia[1,1] += traj.geo.mass_a[a]*(traj.geo.position_ad[a,0]**2 + traj.geo.position_ad[a,2]**2)
        inertia[2,2] += traj.geo.mass_a[a]*(traj.geo.position_ad[a,0]**2 + traj.geo.position_ad[a,1]**2)
        inertia[0,1] -= traj.geo.mass_a[a] *traj.geo.position_ad[a,0] * traj.geo.position_ad[a,1]
        inertia[0,2] -= traj.geo.mass_a[a] *traj.geo.position_ad[a,0] * traj.geo.position_ad[a,2]
        inertia[1,2] -= traj.geo.mass_a[a] *traj.geo.position_ad[a,1] * traj.geo.position_ad[a,2]
    inertia[1][0] = inertia[0][1]
    inertia[2][0] = inertia[0][2]
    inertia[2][1] = inertia[1][2]

    ang_mom = np.zeros(3)
    for a in range(traj.geo.n_atoms):
        mom = traj.geo.velocity_ad[a,:]*traj.geo.mass_a[a]
        L = np.cross(mom, traj.geo.position_ad[a,:])
        ang_mom -= L

    ang_vel = np.linalg.inv(inertia) @ ang_mom
    for a in range(traj.geo.n_atoms):
        v_rot = np.cross(ang_vel, traj.geo.position_ad[a,:])
        traj.geo.velocity_ad[a,:] -= v_rot

def get_kinetic_energy(traj: TrajectorySH):
    kin_energy = 0
    for i in range(traj.geo.n_atoms):
        kin_energy += traj.geo.mass_a[i]*np.inner(traj.geo.velocity_ad[i,:], traj.geo.velocity_ad[i,:])
    return kin_energy/2

def get_total_momentum(traj: TrajectorySH):
    tot_mom = np.zeros(3)
    for a in range(traj.geo.n_atoms):
        for d in range(3):
            tot_mom[d] += traj.geo.velocity_ad[a,d]*traj.geo.mass_a[a]
    return tot_mom

def get_total_ang_momentum(traj: TrajectorySH):
    tot_ang_mom = np.zeros(3)
    for a in range(traj.geo.n_atoms):
        rxv = np.cross(traj.geo.position_ad[a], traj.geo.velocity_ad[a])
        for d in range(3):
            tot_ang_mom[d] += rxv[d]*traj.geo.mass_a[a]
    return tot_ang_mom

def get_momentum_change(traj: TrajectorySH, ctrl: SimulationSH):
    dmom = np.zeros(3)
    for a in range(traj.geo.n_atoms):
        for d in range(3):
            dmom[d] += 0.5*traj.geo.force_sad[traj.hop.active,a,d]*ctrl.dt
    return dmom

def get_ang_momentum_change(traj: TrajectorySH, ctrl: SimulationSH):
    dang_mom = np.zeros(3)
    for a in range(traj.geo.n_atoms):
        rxf = np.cross(traj.geo.position_ad[a], traj.geo.force_sad[traj.hop.active, a])
        for d in range(3):
            dang_mom[d] += 0.5*rxf[d]*ctrl.dt
    return dang_mom

def update_kinematics(traj: TrajectorySH):
    traj.cons.com_position = get_com_position(traj)
    traj.cons.com_velocity = get_com_velocity(traj)

    traj.cons.total_momentum = get_total_momentum(traj)
    traj.cons.total_ang_momentum = get_total_ang_momentum(traj)

    traj.cons.dmomentum[:] = 0.
    traj.cons.dang_momentum[:] = 0.