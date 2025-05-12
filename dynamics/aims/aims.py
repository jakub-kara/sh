import numpy as np
from copy import deepcopy
import shutil
import os
from classes.bundle import Bundle
from classes.molecule import Molecule, MoleculeMixin
from classes.out import Printer
from classes.trajectory import Trajectory
from dynamics.base import Dynamics

class AIMS(Dynamics):
    key = "aims"

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)
        config["nuclear"]["mixins"] = "aims"

        self._csthresh = dynamics.get("csthresh", 0.01)
        self._poptospawn = dynamics.get("poptospawn", 0.01)
        self._omax = dynamics.get("omax", 0.5)
        self._olapthresh = dynamics.get("olapthresh", 1e-3)
        self._regthresh = dynamics.get("regthresh", 1e-4)

    def step_bundle(self, bundle: Bundle):
        super().step_bundle(bundle)

        mol = bundle.active.mol
        if mol.spawn:
            shutil.copytree(f"{bundle.iactive}", f"{bundle.n_traj}", dirs_exist_ok=True)
            with open("events.log", "a") as f:
                temp = np.sum(np.abs(mol.coeff_s[mol.spawn])**2)
                f.write(f"CLONE {bundle.iactive} {np.sqrt(temp)} {bundle.n_traj} {np.sqrt(1-temp)} {bundle.active.timestep.step} {bundle.active.timestep.time:.4f}\n")
            clone = self.spawn_traj(bundle.active)
            os.chdir(f"{bundle.n_traj}")
            clone.write_outputs()
            os.chdir("..")
            bundle.add_trajectory(clone)

    def spawn_traj(self, traj: Trajectory):
        clone = deepcopy(traj)
        traj.mols[-1], clone.mols[-1] = self.spawn_mol(traj.mol)
        traj.mol.spawn = None
        clone.mol.spawn = None
        return clone

    def update_traj(self, traj: Trajectory):
        ts = traj.timestep
        mol = traj.mols[-1]
        mol.phase += 0.5 * mol.kinetic_energy() * ts.dt
        super().update_traj(traj)

        mol = traj.mols[-1]
        mol.phase += 0.5 * mol.kinetic_energy() * ts.dt

    def adjust_nuclear(self, traj: Trajectory):
        mol = traj.mols[-1]
        accbr = self._calculate_breaking(mol)

        for s in range(mol.n_states):
            nac = np.sqrt(np.sum(mol.nacdt_ss[s]**2))
            print(f"{s} {accbr[s]} {nac}")
            if accbr[s] > self._dclone and nac < self._dnac and mol.nspawn < self._maxspawn:
                mol.spawn = [s]
                break

    def spawn_mol(self, mol: Molecule):
        out1 = deepcopy(mol)
        out1.coeff_s[:] = 0
        out1.coeff_s[mol.spawn] = mol.coeff_s[mol.spawn]
        out1.coeff_s /= np.sqrt(np.sum(np.abs(out1.coeff_s)**2))
        self.calculate_acceleration(out1)

        out2 = deepcopy(mol)
        out2.coeff_s[mol.spawn] = 0
        out2.coeff_s /= np.sqrt(np.sum(np.abs(out2.coeff_s)**2))
        self.calculate_acceleration(out2)
        return out1, out2

class AIMSMixin(MoleculeMixin):
    key = "aims"

    def __init__(self, state, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.phase = 0
        self.spawn = None


    def dat_header(self):
        dic = {"phs": Printer.write("Phase", "s")}
        return dic | super().dat_header()

    def dat_dic(self):
        dic = {"phs": Printer.write(self.phase, "f")}
        return dic | super().dat_dic()

    def h5_dict(self):
        dic = {"phase": self.phase}
        return dic | super().h5_dict()