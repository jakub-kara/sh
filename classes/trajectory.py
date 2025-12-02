import numpy as np
import os, pickle
from copy import deepcopy
from .config import Config, get_cfg
from .format import *
from .meta import Counter
from .molecule import Molecule
from .out import FileManager as fm, Timer
from .timestep import Timestep, StateTracker
from electronic.base import ESTProgram
from updaters.nuclear import CompositeIntegrator
from updaters.coeff import CoeffUpdater

@Config("backup", default=True, path=Config.DYNAMICS, valid=Config.BOOL)
class Trajectory:
    def __init__(self, *, dynamics: dict, nuclear: dict, quantum: dict, **config):
        self.index = None
        self._backup = get_cfg(self, "backup", dynamics)

        self.mols: list[Molecule] = []
        self.timestep = None
        self.ref_en = None

        self.track = StateTracker(**dynamics)

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
        self.track.count(self.mol, self.timestep.dt)
        self.timestep.step_success(self.mols)
        self.timestep.save_nupd()

        if self.is_finished:
            print("EST run", Counter.counters["est"], "times")
            self.save_step()

    def add_molecule(self, mol: Molecule):
        self.mols.append(mol)
        return self

    def pop_molecule(self, index: int):
        self.mols.pop(index)
        return self

    def remove_molecule(self, mol: Molecule):
        self.mols.remove(mol)
        return self

    def set_molecules(self, **nuclear):
        nupd = CompositeIntegrator()
        for _ in range(max(nupd.steps, CoeffUpdater().steps, nuclear.get("keep", 2))):
            self.add_molecule(self.mol.copy())

    def set_timestep(self, **dynamics):
        self.timestep: Timestep = get_cfg(Timestep, "timestep", dynamics)(
            keep = self.n_steps, **dynamics)

    def step_header(self):
        fm.log(BORDER)
        fm.log(f"Step:           {fmt(self.timestep.step)}")
        fm.log(f"Time:           {cfmtu(self.timestep.time, 'au', 'fs', Size.S)}")
        fm.log(f"Stepsize:       {cfmtu(self.timestep.dt, 'au', 'fs', Size.S)}")
        fm.log()

    @Timer(id = "save",
           head = "Saving")
    def save_step(self):
        fm.close_all()
        os.chdir("est")
        est = ESTProgram()
        est.backup_wf("../backup/")
        os.chdir("..")

        if self._backup:
            with open("backup/traj.pkl", "wb") as pkl:
                pickle.dump(self, pkl)

    @staticmethod
    def restart(**config):
        with open("backup/traj.pkl", "rb") as pkl:
            traj: Trajectory = pickle.load(pkl)
        traj.restart_components(**config)

        fm.log()
        fm.log(BORDER)
        fm.log("Succesfully restarted from backups.")
        fm.log(BORDER)
        fm.log()
        return traj

    def restart_components(self, *, dynamics: dict, **kwargs):
        self._backup = dynamics.get("backup", True)
        self.timestep.adjust(**dynamics)

    def copy(self):
        return deepcopy(self)

    def energy_diff(self, mol: Molecule, ref: Molecule):
        return np.abs(mol.total_energy() - ref.total_energy())

    def report_energy(self):
        tot = self.mol.total_energy()
        fm.log(f"Total energy:   {cfmtu(tot, 'au', 'ev', Size.M)}")
        fm.log(f"Energy shift:   {cfmtu(self.energy_diff(self.mol, self.mols[-2]), 'au', 'ev', Size.M)}")
        fm.log(f"Energy drift:   {cfmtu(tot - self.ref_en, 'au', 'ev', Size.M)}")
        fm.log()

    def write_headers(self):
        fm.write("dat", self.dat_header())
        fm.write("h5", self.h5_info())

    @Timer(id = "out",
           head = "Outputs")
    def write_outputs(self):
        fm.write("dat", self.dat_dict())
        fm.write("h5", self.h5_dict())
        fm.write("xyz", self.vxyz_string())
        fm.write("dist", self.dist_string())

    def dat_header(self):
        dic = {
            "time": "#" + fmt("Time [fs]")
        }
        return dic | self.mol.dat_header()

    def dat_dict(self):
        dic = {
            "time": cfmt(self.timestep.time, "au", "fs")
        }
        return dic | self.mol.dat_dict()

    def dist_string(self):
        return self.mol.to_dist()

    def xyz_string(self):
        return self.mol.to_xyz()

    def vxyz_string(self):
        return self.mol.to_vxyz()

    def h5_info(self):
        mol = self.mol
        dic = {}
        dic["step"] = "info"
        dic["nst"] = mol.n_states
        dic["nat"] = mol.n_atoms
        dic["ats"] = mol.name_a
        dic["mass"] = mol.mass_a
        return dic

    def h5_dict(self):
        dic = {
            "step": self.timestep.step,
            "time": self.timestep.time
        }
        return dic | self.mol.h5_dict()
