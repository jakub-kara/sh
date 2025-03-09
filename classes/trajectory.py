import numpy as np
import sys, pickle
import time
from copy import deepcopy
from .molecule import Molecule, MoleculeFactory
from .out import Output, Timer
from .constants import convert
from .timestep import Timestep
from electronic.electronic import ESTProgram
from updaters.composite import CompositeIntegrator
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater

class Trajectory:
    def __init__(self, *, dynamics: dict, nuclear: dict, quantum: dict, **config):
        self.index = None
        self._backup = dynamics.get("backup", True)

        self.mols: list[Molecule] = []
        self.timestep = None

    @property
    def n_steps(self):
        return len(self.mols)

    @property
    def mol(self):
        return self.mols[-1]

    @property
    def is_finished(self):
        return self.timestep.finished

    def next_step(self):
        self.timestep.next_step()

    def add_molecule(self, mol: Molecule):
        self.mols.append(mol)
        return self

    def pop_molecule(self, index: int):
        self.mols.pop(index)
        return self

    def remove_molecule(self, mol: Molecule):
        self.mols.remove(mol)
        return self

    def get_molecule(self, **config):
        return MoleculeFactory.create_molecule(n_states=ESTProgram().n_states, **config)

    def set_molecules(self, **nuclear):
        nupd = CompositeIntegrator()
        for _ in range(max(nupd.steps, CoeffUpdater().steps, nuclear.get("keep", 0))):
            self.add_molecule(self.mol.copy_all())

    def set_timestep(self, **dynamics):
        self.timestep: Timestep = Timestep[dynamics.get("timestep", "const")](
            steps = len(self.mols), **dynamics)

    def step_header(self):
        out = Output()
        out.open_log()
        out.write_border()
        out.write_log(f"Step:           {self.timestep.step}")
        out.write_log(f"Time:           {convert(self.timestep.time, 'au', 'fs'):.6f} fs")
        out.write_log(f"Stepsize:       {convert(self.timestep.dt, 'au', 'fs'):.6f} fs")
        out.write_log()

    @Timer(id = "save",
           head = "Saving")
    def save_step(self):
        est = ESTProgram()
        est.backup_wf()

        out = Output()
        out.close_log()
        if self._backup:
            with open("backup/traj.pkl", "wb") as pkl:
                pickle.dump(self, pkl)
        out.open_log()

    @staticmethod
    def restart(**config):
        with open("backup/traj.pkl", "rb") as pkl:
            traj: Trajectory = pickle.load(pkl)
        traj.restart_components(**config)

        out = Output()
        out.open_log()
        out.write_log()
        out.write_border()
        out.write_log("Succesfully restarted from backups.")
        out.write_border()
        out.write_log()
        return traj

    def restart_components(self, *, dynamics: dict, **kwargs):
        self._backup = dynamics.get("backup", True)
        self.timestep.adjust(**dynamics)

    def copy(self):
        return deepcopy(self)
