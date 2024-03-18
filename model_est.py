from classes import Trajectory
import fmodules.models_f as models_f
from abinitio import diagonalise_hamiltonian, adjust_nacmes


def harm(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0,0,0] = 0.5*traj.geo.position_mnad[-1,0,0,0]**2
    traj.pes.ham_diag_mnss[-1,0] = traj.pes.ham_diab_mnss[-1,0]
    traj.pes.nac_ddr_mnssad[-1,0,0,0] = traj.geo.position_mnad[-1,0]

def spin_boson(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], gradH_ssad = models_f.spin_boson(traj.geo.position_mnad[-1, 0])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, 0] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, 0], traj.pes.ham_transform_mnss[-1, 0], gradH_ssad)
    #adjust_nacmes(traj)

def tully_1(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], gradH_ssad = models_f.tully_1(traj.geo.position_mnad[-1, 0])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, 0] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, 0], traj.pes.ham_transform_mnss[-1, 0], gradH_ssad)
    #adjust_nacmes(traj)

def tully_s(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], gradH_ssad = models_f.tully_s(traj.geo.position_mnad[-1, 0])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, 0] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, 0], traj.pes.ham_transform_mnss[-1, 0], gradH_ssad)
    adjust_nacmes(traj)

def tully_n(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], gradH_ssad = models_f.tully_n(traj.geo.position_mnad[-1, 0])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, 0] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, 0], traj.pes.ham_transform_mnss[-1, 0], gradH_ssad)
    adjust_nacmes(traj)

def tully_2(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], gradH_ssad = models_f.tully_2(traj.geo.position_mnad[-1, 0])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, 0] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, 0], traj.pes.ham_transform_mnss[-1, 0], gradH_ssad)
    adjust_nacmes(traj)

def tully_3(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], gradH_ssad = models_f.tully_3(traj.geo.position_mnad[-1, 0])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, 0] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, 0], traj.pes.ham_transform_mnss[-1, 0], gradH_ssad)
    adjust_nacmes(traj)

def sub_x(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], gradH_ssad = models_f.sub_x(traj.geo.position_mnad[-1, 0])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, 0] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, 0], traj.pes.ham_transform_mnss[-1, 0], gradH_ssad)
    adjust_nacmes(traj)

def sub_s(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], gradH_ssad = models_f.sub_s(traj.geo.position_mnad[-1, 0])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, 0] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, 0], traj.pes.ham_transform_mnss[-1, 0], gradH_ssad)
    adjust_nacmes(traj)

def sub_2(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], gradH_ssad = models_f.sub_2(traj.geo.position_mnad[-1, 0])
    diagonalise_hamiltonian(traj)
    traj.pes.nac_ddr_mnssad[-1, 0] = models_f.get_nac_and_gradient(traj.pes.ham_diag_mnss[-1, 0], traj.pes.ham_transform_mnss[-1, 0], gradH_ssad)
    adjust_nacmes(traj)

def lvc_wrapper(traj: Trajectory):
    traj.pes.ham_diab_mnss[-1,0], traj.pes.ham_diag_mnss[-1,0], traj.pes.ham_transform_mnss[-1,0], traj.pes.nac_ddr_mnssad[-1,0] = \
        LVC.get_est(traj.geo.position_mnad[-1,0].flatten())