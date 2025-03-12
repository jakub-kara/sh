import numpy as np
import os, sys, shutil
import pickle
from copy import deepcopy
from .meta import Singleton
from .molecule import MoleculeFactory
from .out import Output
from classes.trajectory import Trajectory
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram
from updaters.composite import CompositeIntegrator
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater

class Bundle:
    def __init__(self):
        self._trajs: list[Trajectory] = []
        self._iactive: int = 0
        self._active: Trajectory = None

    def save_setup(self):
        with open("single.pkl", "wb") as pkl:
            pickle.dump(Singleton.save(), pkl)

        with open("mol.pkl", "wb") as pkl:
            pickle.dump(MoleculeFactory.save(), pkl)

    @staticmethod
    def load_setup():
        with open("single.pkl", "rb") as pkl:
            Singleton.restart(pickle.load(pkl))

        with open("mol.pkl", "rb") as pkl:
            MoleculeFactory.restart(pickle.load(pkl))

    @staticmethod
    def restart(**config):
        bundle = Bundle()
        traj_dirs = [d for d in os.listdir() if (os.path.isdir(d) and d.isdigit())]
        Bundle.load_setup()
        for d in traj_dirs:
            os.chdir(d)
            traj = Trajectory.restart(**config)
            bundle.add_trajectory(traj)
            os.chdir("..")
        return bundle

    @property
    def n_traj(self):
        return len(self._trajs)

    def add_trajectory(self, traj: Trajectory):
        traj.index = self.n_traj
        self._trajs.append(traj)
        return self

    def set_active(self):
        self._iactive = np.argmin([traj.timestep.time for traj in self._trajs])
        self._active = self._trajs[self._iactive]
        return self

    def setup(self, dynamics, **config):
        config["nuclear"].setdefault("mixins", [])
        self.set_dynamics(dynamics, **config)
        self.set_components(**config)

        # TODO: refactor
        traj = Trajectory(dynamics=dynamics, **config)
        mol = traj.get_molecule(**config["nuclear"], **dynamics)
        Dynamics().read_coeff(mol, config["quantum"].get("input", None))
        traj.add_molecule(mol)
        traj.set_molecules(**config["nuclear"])
        traj.set_timestep(**dynamics)
        self.save_setup()
        self.add_trajectory(traj)

        with open("events.log", "w") as f:
            f.write(f"INIT 0\n")
        self.prepare_trajs()
        return self

    def set_components(self, *, electronic, nuclear, quantum, output, **kwargs):
        self.set_est(**electronic)
        self.set_nuclear(**nuclear)
        self.set_tdc_updater(**quantum)
        self.set_coeff_updater(**quantum)
        self.set_io(**output)

    def set_dynamics(self, dynamics, **config):
        Dynamics.reset()
        Dynamics[dynamics["method"]](dynamics=dynamics, **config)

    def set_est(self, **electronic):
        ESTProgram.reset()
        ESTProgram[electronic["program"]](**electronic)

    def set_nuclear(self, **nuclear):
        CompositeIntegrator.reset()
        CompositeIntegrator(nuc_upd = nuclear["nuc_upd"])

    def set_tdc_updater(self, **quantum):
        TDCUpdater.reset()
        TDCUpdater[quantum["tdc_upd"]](**quantum)

    def set_coeff_updater(self, **quantum):
        CoeffUpdater.reset()
        CoeffUpdater[quantum["coeff_upd"]](**quantum)

    def set_io(self, **output):
        Output(**output)

    def prepare_trajs(self):
        out = Output()
        for traj in self._trajs:
            os.chdir(f"{traj.index}")
            out.open_log(mode="w")
            out.to_log("../" + sys.argv[1])
            Dynamics().prepare_traj(traj)
            os.chdir("..")

    def run_step(self):
        self.set_active()
        print()
        if self.n_traj > 1:
            print(self._iactive, self.n_traj)

        os.chdir(f"{self._iactive}")
        Dynamics().run_step(self._active)
        os.chdir("..")

        mol = self._active.mol
        if hasattr(mol, "split") and mol.split:
            shutil.copytree(f"{self._iactive}", f"{self.n_traj}", dirs_exist_ok=True)
            with open("events.log", "a") as f:
                temp = np.sum(np.abs(self._active.mol.coeff_s[self._active.split])**2)
                f.write(f"CLONE {self._iactive} {np.sqrt(temp)} {self.n_traj} {np.sqrt(1-temp)} {self._active.timestep.step} {self._active.timestep.time:.4f}\n")
            clone = self.split_traj(self._active)
            self.add_trajectory(clone)
        return self


    def split_traj(self, traj: Trajectory):
        clone = deepcopy(traj)
        dyn = Dynamics()
        traj.mols[-1], clone.mols[-1] = dyn.split_mol(traj.mol)
        traj.mol.split = None
        clone.mol.split = None
        return clone

    @property
    def is_finished(self):
        return np.all([traj.is_finished for traj in self._trajs])

    def edit(self, attr, val):
        for traj in self._trajs:
            setattr(traj, attr, val)