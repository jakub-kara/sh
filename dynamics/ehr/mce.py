import numpy as np
from copy import deepcopy
import shutil
import os
from .ehr import SimpleEhrenfest, EhrMixin
from classes.bundle import Bundle
from classes.molecule import Molecule
from classes.out import Printer
from classes.trajectory import Trajectory

class MultiEhrenfest(SimpleEhrenfest):
    key = "mce"

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)
        config["nuclear"]["mixins"] = "mce"

        self._dclone = dynamics.get("dclone", 5e-6)
        self._dnac = dynamics.get("dnac", 2e-3)
        self._maxspawn = dynamics.get("maxspawn", 3)

        self._clone = None
        self._trans = 0

    def _calculate_breaking(self, mol: Molecule):
        nst = mol.n_states
        accbr = np.zeros(nst)
        for s in range(mol.n_states):
            dfbr = mol.grad_sad[s] + mol.acc_ad * mol.mass_a[:,None]
            fbr = np.abs(mol.coeff_s[s])**2 * dfbr
            accbr[s] = np.linalg.norm(fbr / mol.mass_a[:,None])
        return accbr

    def step_bundle(self, bundle: Bundle):
        super().step_bundle(bundle)

        if self._clone:
            shutil.copytree(f"{bundle.iactive}", f"{bundle.n_traj}", dirs_exist_ok=True)
            with open("events.log", "a") as f:
                f.write(f"CLONE {bundle.iactive} {np.sqrt(self._trans)} {bundle.n_traj} {np.sqrt(1-self._trans)} {bundle.active.timestep.step} {bundle.active.timestep.time:.4f}\n")


            os.chdir(f"{bundle.n_traj}")
            self._clone.next_step()
            self._clone.write_outputs()
            os.chdir("..")
            bundle.add_trajectory(self._clone)
            self._clone = None

    def spawn_traj(self, traj: Trajectory, state: int):
        clone = deepcopy(traj)
        print(np.abs(traj.mol.coeff_s)**2)
        traj.mols[-1], clone.mols[-1] = self.spawn_mol(traj.mol, state)
        print(np.abs(traj.mol.coeff_s)**2)
        print(np.abs(clone.mol.coeff_s)**2)
        self._clone = clone

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
                mol.nspawn += 1
                self.spawn_traj(traj, s)
                break

    def spawn_mol(self, mol: Molecule, state: int):
        self._trans = np.sum(np.abs(mol.coeff_s[state])**2)

        out1 = deepcopy(mol)
        out1.coeff_s[:] = 0
        out1.coeff_s[state] = mol.coeff_s[state]
        out1.coeff_s /= np.sqrt(np.sum(np.abs(out1.coeff_s)**2))
        self.calculate_acceleration(out1)

        out2 = deepcopy(mol)
        out2.coeff_s[state] = 0
        out2.coeff_s /= np.sqrt(np.sum(np.abs(out2.coeff_s)**2))
        self.calculate_acceleration(out2)
        return out1, out2

class MCEMixin(EhrMixin):
    key = "mce"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nspawn = 0
        self.phase = 0

    def dat_header(self):
        dic = {"phs": Printer.write("Phase", "s")}
        return dic | super().dat_header()

    def dat_dict(self):
        dic = {"phs": Printer.write(self.phase, "f")}
        return dic | super().dat_dict()

    def h5_dict(self):
        dic = {"phase": self.phase}
        return dic | super().h5_dict()