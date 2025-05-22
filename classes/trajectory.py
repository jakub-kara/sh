import numpy as np
import sys, pickle
from copy import deepcopy
from .meta import Counter
from .molecule import Molecule
from .out import Output as out, Timer, Printer
from .constants import convert
from .timestep import Timestep, StateTracker
from electronic.base import ESTProgram
from updaters.composite import CompositeIntegrator
from updaters.coeff import CoeffUpdater

class Trajectory:
    def __init__(self, *, dynamics: dict, nuclear: dict, quantum: dict, **config):
        self.index = None
        self._backup = dynamics.get("backup", True)

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
        self.timestep.step_success()
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
        for _ in range(max(nupd.steps, CoeffUpdater().steps, nuclear.get("keep", 0))):
            self.add_molecule(self.mol.copy_all())

    def set_timestep(self, **dynamics):
        self.timestep: Timestep = Timestep.select(dynamics.get("timestep", "const"))(
            steps = len(self.mols), **dynamics)


    def step_header(self):
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

        if self._backup:
            with open("backup/traj.pkl", "wb") as pkl:
                pickle.dump(self, pkl)

    @staticmethod
    def restart(**config):
        with open("backup/traj.pkl", "rb") as pkl:
            traj: Trajectory = pickle.load(pkl)
        traj.restart_components(**config)

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

    def energy_diff(self, mol: Molecule, ref: Molecule):
        return np.abs(mol.total_energy() - ref.total_energy())

    def report_energy(self):
        tot = self.mol.total_energy()
        out.write_log(f"Total energy:   {convert(tot, 'au', 'ev'):.6f} eV")
        out.write_log(f"Energy shift:   {convert(self.energy_diff(self.mol, self.mols[-2]), 'au', 'ev'):.6f} eV")
        out.write_log(f"Energy drift:   {convert(tot - self.ref_en, 'au', 'ev'):.6f} eV")
        out.write_log()

    def write_headers(self):
        out.write_dat(self.dat_header(), "w")
        out.write_h5(self.h5_info(), "w")
        out.write_xyz("", "w")
        out.write_dist("", "w")

    @Timer(id = "out",
           head = "Outputs")
    def write_outputs(self):
        out.write_dat(self.dat_dict())
        out.write_h5(self.h5_dict())
        out.write_xyz(self.vxyz_string())
        out.write_dist(self.dist_string())

    def dat_header(self):
        dic = {
            "time": "#" + Printer.write("Time [fs]", "s")
        }
        return dic | self.mol.dat_header()

    def dat_dict(self):
        dic = {
            "time": Printer.write(convert(self.timestep.time, "au", "fs"), "f")
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
