import numpy as np

from constants import Constants
from utility import get_dict_value, read_initial_conditions

class TrajectorySH:
    def __init__(self, input_dict: dict):
        self.id = get_dict_value(input_dict["general"], "id", "0")
        self.name = get_dict_value(input_dict["general"], "name", "name")

        self.io = IO(self.id, input_dict)
        self.geo = Geometry(input_dict)
        if get_dict_value(input_dict["electronic"], "type") == "casscf":
            self.est = ElectronicStructure(input_dict, self.geo.n_atoms)
        else:
            self.est = ModelPotential(input_dict, self.geo.n_atoms)
        self.hop = Hopping(input_dict, self.est.n_states)
        self.cons = Conservations()

class IO:
    def __init__(self, id: str, input_dict: dict):
        self.xyz_file = f"data/traj_{id}.xyz"
        self.dat_file = f"data/traj_{id}.dat"
        self.record = get_dict_value(input_dict["general"], "record", "").split(",")

class Geometry:
    def __init__(self, input_dict: dict):
        input_file = get_dict_value(input_dict["nuclear"], "input")
        self.n_atoms, self.position_ad, self.velocity_ad, self.name_a, self.mass_a = read_initial_conditions(input_file)
        n_states = int(get_dict_value(input_dict["electronic"], "nstates"))
        self.force_sad = np.zeros((n_states, self.n_atoms, 3), order='F')

class ModelPotential:
    def __init__(self, input_dict: dict, n_atoms: int):
        self.type = get_dict_value(input_dict["electronic"], "type")
        self.n_states = int(get_dict_value(input_dict["electronic"], "nstates"))

        self.recalculate = False

        self.reference_energy = 0.

        self.pes = PotentialEnergySurface(self.n_states, n_atoms)
        self.pes_old = self.pes

class ElectronicStructure:
    def __init__(self, input_dict: dict, n_atoms: int):
        self.program = "molpro"
        self.type = get_dict_value(input_dict["electronic"], "type")
        self.n_states = int(get_dict_value(input_dict["electronic"], "nstates"))
        self.file = f"{self.program}_"

        self.active_orb = int(get_dict_value(input_dict["electronic"], "active"))
        self.closed_orb = int(get_dict_value(input_dict["electronic"], "closed"))
        self.n_el = int(get_dict_value(input_dict["electronic"], "nelectrons"))

        self.civec = get_dict_value(input_dict["electronic"], "civec", "false") in Constants.true
        self.first = get_dict_value(input_dict["electronic"], "first", "true") in Constants.true
        self.molden = get_dict_value(input_dict["electronic"], "molden", "false") in Constants.true
        self.basis = get_dict_value(input_dict["electronic"], "basis", "avdz")

        self.density_fit = get_dict_value(input_dict["electronic"], "df", "false") in Constants.true

        self.recalculate = False
        self.reference_energy = 0.

        self.pes = PotentialEnergySurface(self.n_states, n_atoms)
    
class PotentialEnergySurface:
    def __init__(self, n_states: int, n_atoms: int): 
        self.ham_diab_ss = np.zeros((n_states, n_states), dtype=np.complex128)
        self.ham_diag_ss = np.zeros((n_states, n_states), dtype=np.complex128)
        self.ham_transform_ss = np.zeros((n_states, n_states), dtype=np.complex128)
        self.diagonalise = True
        self.nac_ddr_ssad = np.zeros((n_states, n_states, n_atoms, 3), order='F')
        self.nac_ddt_ss = np.zeros((n_states, n_states), order='F')
        self.nac_flip = np.zeros((n_states, n_states), dtype=bool)

class Hopping:
    def __init__(self, input_dict: dict, n_states: int):
        self.active = int(get_dict_value(input_dict["hopping"], "active_state", " "))
        self.seed = np.random.seed()

        self.state_coeff_s = np.zeros(n_states, dtype=np.complex128)
        self.state_coeff_s[self.active] = 1.
        self.prob_s = np.zeros(n_states)
        self.target = self.active

class Conservations:
    def __init__(self):
        self.potential_energy = 0.
        self.com_position = np.zeros(3)
        self.com_velocity = np.zeros(3)
        self.total_momentum = np.zeros(3)
        self.dmomentum = np.zeros(3)
        self.total_ang_momentum = np.zeros(3)
        self.dang_momentum = np.zeros(3)

class SimulationSH:
    def __init__(self, input_dict: dict):
        self.dt_max = float(get_dict_value(input_dict["nuclear"], "dt_max", " "))
        self.dt = self.dt_max
        self.iter = 0
        self.max_iter = 10
        self.quantum_resolution = int(get_dict_value(input_dict["hopping"], "quantum_res", " "))
        self.dtq = self.dt/self.quantum_resolution
        
        self.current_time = 0
        self.step = 0
        self.qstep = 0

        self.termination_cond = get_dict_value(input_dict["nuclear"], "termination", " ")

        """
        self.seed = 2
        np.random.seed(seed=self.seed)
        """