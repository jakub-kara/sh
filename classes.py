import numpy as np
from typing import Callable

from errors import *
from constants import Constants
from utility import read_initial_conditions


# MAKE CONSTANTS CLASS
# INCLUDE OFFSET FOR STATES
class Trajectory:
    def __init__(self, config: dict):
        self.par = Params(config)
        self.ctrl = Control(config, self.par)
        self.io = IO(config, self.par)
        self.geo = Geometry(config, self.par)
        self.est = ElectronicStructure(config, self.par)
        self.pes = PotentialEnergySurface(config, self.par)
        self.hop = Hopping(config, self)

class Params:
    def __init__(self, config: dict):
        self.id = 0
        self.name = config["control"]["name"]
        self.type = config["control"]["trajtype"]

        self.n_steps = 5
        self.n_substeps = 1
        temp = np.trim_zeros(np.array(config["electronic"]["nstates"]), "b")
        self.states = temp*(np.arange(temp.shape[0]) + 1)
        self.n_states = np.sum(self.states)
        with open("geom.xyz", "r") as geomfile: self.n_atoms = int(geomfile.readline().strip())
        self.n_qsteps = config["control"]["qres"]

        self.ref_en = 0.

class IO:
    def __init__(self, config: dict, par: Params):
        self.xyz_file = f"data/traj_{par.id}.xyz"
        self.dat_file = f"data/traj_{par.id}.dat"
        self.log_file = f"data/traj_{par.id}.log"
        self.mat_file = f"data/traj_{par.id}.mat"

        self.record = config["control"]["record"]

class Geometry:
    def __init__(self, config: dict, par: Params):
        self.position_mnad = np.zeros((par.n_steps, par.n_substeps, par.n_atoms, 3))
        self.velocity_mnad = np.zeros((par.n_steps, par.n_substeps, par.n_atoms, 3))
        self.force_mnad = np.zeros((par.n_steps, par.n_substeps, par.n_atoms, 3))

        input_file = "geom.xyz"
        _, position_ad, velocity_ad, self.name_a, self.mass_a = read_initial_conditions(input_file)
        self.position_mnad[-1,0] = position_ad
        self.velocity_mnad[-1,0] = velocity_ad

        self.scheme_name = config["nuclear"]["integrator"]
        self.init_solver, self.init_scheme, self.loop_solver, self.loop_scheme = None, None, None, None
        self.force_updater: Callable[[Trajectory], None] = None
    
class ElectronicStructure:
    def __init__(self, config: dict, par: Params):
        self.program = config["electronic"]["program"]
        self.type = config["electronic"]["esttype"]
        self.run: Callable[[Trajectory], None] = None
        self.file = ""
        self.first = True
        self.skip = config["electronic"]["skip"]

        self.tdc_updater = config["electronic"]["tdc"]

        self.calculate_nacs = np.zeros((par.n_states, par.n_states))
        self.nacs_setter = Callable[[Trajectory], None]
        self.coeff_mns = np.zeros((par.n_steps, par.n_substeps, par.n_states), dtype=np.complex128)
        self.propagator_name = config["electronic"]["propagator"]
        self.propagator: Callable[[Trajectory], None] = None

        self.config: dict = config["electronic"].get("config", None)
        
        # CASPT2

        """self.caspt2 = config.electronic.caspt2.value
        self.imag = config.electronic.imag.value
        self.shift = config.electronic.shift.value
        self.caspt2_type = config.electronic.caspt2_type.value
        """
        
class PotentialEnergySurface:
    def __init__(self, config: dict, par: Params):
        self.ham_diab_mnss = np.zeros((par.n_steps, par.n_substeps, par.n_states, par.n_states), dtype=np.complex128)
        self.ham_diag_mnss = np.zeros((par.n_steps, par.n_substeps, par.n_states, par.n_states))
        self.poten_mn = np.zeros((par.n_steps, par.n_substeps))
        self.ham_transform_mnss = np.zeros((par.n_steps, par.n_substeps, par.n_states, par.n_states), dtype=np.complex128)
        self.diagonalise = True
        self.nac_ddr_mnssad = np.zeros((par.n_steps, par.n_substeps, par.n_states, par.n_states, par.n_atoms, 3))
        self.nac_ddt_mnss = np.zeros((par.n_steps, par.n_substeps, par.n_states, par.n_states))
        self.overlap_mnss = np.zeros((par.n_steps, par.n_substeps, par.n_states, par.n_states))
        self.nac_flip = np.zeros((par.n_states, par.n_states), dtype=bool)
        self.phase_s = np.ones(par.n_states)

        self.order = np.zeros(par.n_states, dtype=int)

class Hopping:
    def __init__(self, config: dict, traj: Trajectory):
        self.seed = np.random.seed()
        self.active = config["electronic"]["initstate"] - config["electronic"]["skip"]

        self.type = config["hopping"]["shtype"]
        self.decoherence = config["hopping"]["decoherence"]


        # REWORK DEPENDENCIES
        if self.type == "mash":
            self.rescale = "mash"
            temp = np.genfromtxt("coeff")
            traj.est.coeff_mns[-1,0] = temp[:,0] + 1j * temp[:,1]
        elif self.type == "fssh":
            self.rescale = "ddr"
            traj.est.coeff_mns[-1,0,self.active] = 1.
            if traj.est.tdc_updater != 'nacme':
                print('setting rescale to velocity')
                self.rescale = ''
        else:
            raise HoppingTypeNotFoundError

        #create population check!!

        self.prob_s = np.zeros(traj.par.n_states)
        self.target = self.active

class Control:
    def __init__(self, config: dict, par: Params):
        self.substep = 0
        self.qstep = 0
        self.init_steps = par.n_steps
        self.conv_status = 0
        self.en_thresh = np.zeros(3)
        # 1 value => all three thresholds are the same
        if len(config["control"]["enthresh"]) == 1:
            self.en_thresh[:] = config["control"]["enthresh"][0]
        # 2 values => RKN thresholds are the same
        elif len(config["control"]["enthresh"]) == 2:
            self.en_thresh[:2] = config["control"]["enthresh"][0]
            self.en_thresh[2] = config["control"]["enthresh"][1]
        # 3 values => 3 different thresholds
        elif len(config["control"]["enthresh"]) == 3:
            self.en_thresh[:] = config["control"]["enthresh"]
        self.en_thresh /= Constants.eh2ev

        temp = 1 if config["control"]["tunit"] == "au" else 1/Constants.au2fs
        self.t_max = config["control"]["tmax"] * temp
        self.dt_func: Callable[[Trajectory, float], float] = None
        if config["control"]["adapt"]:
            self.dt_name = config["control"]["stepfunc"]
            self.dt_max = config["control"]["stepmax"] * temp
            self.dt_min = config["control"]["stepmin"] * temp
            self.dt_params = config["control"]["stepparams"]
        else:
            self.dt_name = "const"
            self.dt_max = config["control"]["step"] * temp
        self.dt = 0
        self.h = np.zeros(par.n_steps)
        self.dtq = self.dt/par.n_qsteps
        
        self.curr_time = 0
        self.curr_step = 0

        self.timing = np.zeros(7)

        self.seed = config["control"].get("seed", 0)
        np.random.seed(seed=self.seed)
