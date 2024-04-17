import numpy as np
from typing import Callable

from errors import *
from constants import Constants
from utility import read_initial_conditions

class Trajectory:
    """
    Stores all the information about a single trajectory.
    Except for constructors, there are no instance methods present.
    Rather, methods are called externally in resemblance to C-like structs.
    """
    
    def __init__(self, config: dict):
        """
        Constructs the Trajectory class and all its subclasses.
        All the parameters are contained in the provided dictionary.
        
        Parameters
        ----------
        config: dict
            file speifying all the information needed to initialise the trajctory
        
        Returns
        -------
        None
        
        Modifies
        --------
        self
        """
        
        # contains parameters of the trajectory, ie unchanging quantities
        self.par = Params(config)
        # contains quantities and methods that control the simulation
        self.ctrl = Control(config, self.par)
        # contains input/output information
        self.io = IO(config, self.par)
        # contains all quantities related to geometries
        # the size of the array is (n_steps, n_substeps)
        self.geo_mn = np.array([[Geometry(config, self.par) for n in range(self.par.n_substeps)] for m in range(self.par.n_steps)], dtype=np.dtype(Geometry))
        # set the initial position, velocity, atom masses and names
        input_file = "geom.xyz"
        _, self.geo_mn[-1,0].position_ad, self.geo_mn[-1,0].velocity_ad, self.geo_mn[-1,0].name_a, self.geo_mn[-1,0].mass_a = read_initial_conditions(input_file)
        # contains all quantities related to potential energy surfaces
        # the size of the array is (n_steps, n_substeps)
        self.pes_mn = np.array([[PotentialEnergySurface(config, self.par) for n in range(self.par.n_substeps)] for m in range(self.par.n_steps)])
        # contains all parameters related to electronic structure calculations
        self.est = ElectronicStructure(config, self.par)
        # special modules based on the type of trajectory
        if self.par.type == "sh":
            self.hop = Hopping(config, self)

class Params:
    """
    A class that contains static information about the trajectory.
    """
    
    def __init__(self, config: dict):
        """
        Construct the Params class.
        All the parameters are contained in the provided dictionary.
        
        Parameters
        ----------
        config: int
            file specifying all the information needed to initialise the trajectory
        
        Returns
        -------
        None
        
        Modifies
        --------
        self
        """
        
        # TODO: make customisable
        self.id = 0
        # system name
        self.name = config["control"]["name"]
        # type of trajectory [SH/MFE]
        self.type = config["control"]["trajtype"]

        # number of steps (previous + current) to be temporarily saved
        self.n_steps = 5
        # number of substeps (integrator-dependent) to be saved
        self.n_substeps = 1
        # number of states in each spin manifold without considering spin degeneracy
        temp = np.trim_zeros(np.array(config["electronic"]["nstates"]), "b")
        # number of states in each spin manifold considering spin degeneracy
        self.states = temp*(np.arange(temp.shape[0]) + 1)
        # total number of states
        self.n_states = np.sum(self.states)
        # number of atoms
        with open("geom.xyz", "r") as geomfile: self.n_atoms = int(geomfile.readline().strip())
        # number of quantum steps per classical step
        self.n_qsteps = config["control"]["qres"]
        # reference potential energy
        self.ref_en = 0.

class IO:
    """
    A class that contains all the input/output information.
    """
    def __init__(self, config: dict, par: Params):
        """
        Construct the IO class.
        All the parameters are contained in the provided dictionary and Params instance.
        
        Parameters
        ----------
        config: int
            file specifying all the information needed to initialise the trajectory
        par: Params
            already initialised Params object
        
        Returns
        -------
        None
        
        Modifies
        --------
        self
        """
        
        self.xyz_file = f"data/traj_{par.id}.xyz"
        self.dat_file = f"data/traj_{par.id}.dat"
        self.log_file = f"data/traj_{par.id}.log"
        self.mat_file = f"data/traj_{par.id}.mat"

        self.record = config["control"]["record"]

class Geometry:
    """
    A class that contains all the geometry information.
    """
    def __init__(self, config: dict, par: Params):
        """
        Construct the Geometry class.
        All the parameters are contained in the provided dictionary and Params instance.
        
        Parameters
        ----------
        config: int
            file specifying all the information needed to initialise the trajectory
        par: Params
            already initialised Params object
        
        Returns
        -------
        None
        
        Modifies
        --------
        self
        """
                
        self.position_ad = np.zeros((par.n_atoms, 3))
        self.velocity_ad = np.zeros((par.n_atoms, 3))
        self.force_ad = np.zeros((par.n_atoms, 3))
        self.name_a = np.full(par.n_atoms, "000")
        self.mass_a = np.zeros(par.n_atoms)

class PotentialEnergySurface:
    """
    A class that contains all the information about the potential energy surface.
    """
    def __init__(self, config: dict, par: Params):
        """
        Construct the PotentialEnergySurface class.
        All the parameters are contained in the provided dictionary and Params instance.
        
        Parameters
        ----------
        config: int
            file specifying all the information needed to initialise the trajectory
        par: Params
            already initialised Params object
        
        Returns
        -------
        None
        
        Modifies
        --------
        self
        """

        # spin-diabatic hamiltonian (identical to adiabatic hamiltonian without soc)
        self.ham_diab_ss = np.zeros((par.n_states, par.n_states), dtype=np.complex128)
        # diagonal hamiltonian
        self.ham_diag_ss = np.zeros((par.n_states, par.n_states))
        # potential energy
        self.poten = 0.
        # unitary transformation between diabatic and diagonal hamiltonian
        self.ham_transform_ss = np.zeros((par.n_states, par.n_states), dtype=np.complex128)
        # on-diagonal: gradients
        # off-diagonal: nonadiabatic coupling matrix elements
        self.nac_ddr_ssad = np.zeros((par.n_states, par.n_states, par.n_atoms, 3))
        # time-derivative coupling matrix elements
        self.nac_ddt_ss = np.zeros((par.n_states, par.n_states))
        # wf overlap between current and previous timestep
        self.overlap_ss = np.zeros((par.n_states, par.n_states))
        # dipole moment
        self.dip_mom_ssd = np.zeros((par.n_states, par.n_states, 3))
        # nacme phase correction
        self.nac_flip_ss = np.zeros((par.n_states, par.n_states), dtype=bool)
        # tdc phase
        self.phase_s = np.ones(par.n_states)

class ElectronicStructure:
    """
    A class that contains all the information about electronic structure calculations.
    """
    def __init__(self, config: dict, par: Params):
        """
        Construct the ElectronicStructure class.
        All the parameters are contained in the provided dictionary and Params instance.
        
        Parameters
        ----------
        config: int
            file specifying all the information needed to initialise the trajectory
        par: Params
            already initialised Params object
        
        Returns
        -------
        None
        
        Modifies
        --------
        self
        """

        # name of the EST program to be used in the calculations
        self.program = config["electronic"]["program"]
        # type of EST calculation/level of theory
        self.type = config["electronic"]["esttype"]
        # Path to run
        self.path = config["electronic"]["programpath"]
        # method that interfaces EST
        self.run: Callable[[Trajectory], None] = None
        # root/name of the EST files
        self.file = f"{self.program}"
        # whether this is the first EST calculation
        self.first = True
        # if some states are to be skipped [not saved and used for transformations]
        self.skip = config["electronic"]["skip"]

        # which gradients/nacmes to calculate
        self.calc_nacs = np.zeros((par.n_states, par.n_states))
        # method to request nacme calculations
        self.nacs_setter = Callable[[Trajectory], None]
        # wavefunction coefficient
        self.coeff_mns = np.zeros((par.n_steps, par.n_substeps, par.n_states), dtype=np.complex128)

        # additional config for EST
        self.config: dict = config["electronic"].get("config", None)

class Hopping:
    """
    A class that contains additional information for surface-hopping trajectories.
    """
    def __init__(self, config: dict, traj: Trajectory):
        """
        Construct the Hopping class.
        All the parameters are contained in the provided dictionary and Trajectory instance.
        
        Parameters
        ----------
        self: Params
            Params object
        config: int
            file specifying all the information needed to initialise the trajectory
        par: Trajectory
            already initialised Trajectory object
        
        Returns
        -------
        None
        
        Modifies
        --------
        self
        """

        # random seed for the trajectory
        self.seed = np.random.seed()
        # initial active state
        self.active = config["electronic"]["initstate"] - config["electronic"]["skip"]
        # type of surface hopping
        self.type = config["hopping"]["shtype"]
        # decoherence correction
        self.decoherence = config["hopping"]["decoherence"]

        # TODO: REWORK DEPENDENCIES
        # MASH style WF coefficient initialisation
        if self.type == "mash":
            # how to rescale velocity when hopping
            self.rescale = "mash"
            temp = np.genfromtxt("coeff")
            traj.est.coeff_mns[-1,0] = temp[:,0] + 1j * temp[:,1]
        # FSSH style WF coefficient initialisation
        elif self.type == "fssh":
            # project velocity along nacmes
            self.rescale = "ddr"
            # set the WF coefficient to unity on the active state
            traj.est.coeff_mns[-1,0,self.active] = 1.
            if traj.ctrl.tdc_updater != 'nacme':
                print('setting rescale to velocity')
                self.rescale = ''
        else:
            raise HoppingTypeNotFoundError

        #create population check!!

        # hopping probabilities
        self.prob_s = np.zeros(traj.par.n_states)
        # candidate state for transition
        self.target = self.active

class Control:
    """
    A class that control the running of the simulation.
    """
    def __init__(self, config: dict, par: Params):
        """
        Construct the Control class.
        All the parameters are contained in the provided dictionary and Params instance.
        
        Parameters
        ----------
        self: Params
            Params object
        config: int
            file specifying all the information needed to initialise the trajectory
        par: Params
            already initialised Params object
        
        Returns
        -------
        None
        
        Modifies
        --------
        self
        """

        # substep number
        self.substep = 0
        # quantum step number
        self.qstep = 0
        # energy convergence flag
        self.conv_status = 0
        # energy thresholds
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

        # unit conversion for time-like input
        temp = 1 if config["control"]["tunit"] == "au" else 1/Constants.au2fs
        # duration of the simulation
        self.t_max = config["control"]["tmax"] * temp
        # method that determines the current stepsize
        self.dt_func: Callable[[Trajectory, float], float] = None
        # settings for adaptive stepsize
        if config["control"]["adapt"]:
            self.dt_name = config["control"]["stepfunc"]
            self.dt_max = config["control"]["stepmax"] * temp
            self.dt_min = config["control"]["stepmin"] * temp
            self.dt_params = config["control"]["stepparams"]
        # fixed stepsize
        else:
            self.dt_name = "const"
            self.dt_max = config["control"]["step"] * temp
        self.dt = 0
        # previous stepsizes
        self.h = np.zeros(par.n_steps)
        self.dtq = self.dt/par.n_qsteps
        
        # time elapsed since the start of simulation
        self.curr_time = 0
        # classical step number
        self.curr_step = 0

        # nuclear integrator
        self.nuc_scheme_name = config["nuclear"]["integrator"]
        self.nuc_init_scheme, self.nuc_loop_scheme = None, None
        # number of initialisation steps needed for the chosen integrator
        self.init_steps = 0
        self.force_updater: Callable[[Trajectory], None] = None

        # WF coefficient integrator
        self.wfc_scheme_name = config["electronic"]["propagator"]
        self.wfc_prop: Callable[[Trajectory], None] = None

        # scheme for updating tdc
        self.tdc_updater = config["electronic"]["tdc"]
        # should the hamiltonian be diagonalised
        self.diagonalise = True

        # set seed
        self.seed = config["control"].get("seed", 0)
        np.random.seed(seed=self.seed)
