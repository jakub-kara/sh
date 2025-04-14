import numpy as np
from .ehr import SimpleEhrenfest
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

    # TODO: symmetrise breaking force
    def _calculate_breaking(self, mol: Molecule):
        nst = mol.n_states
        accbr = np.zeros(nst)
        for s in range(mol.n_states):
            dfbr = mol.grad_sad[s] + mol.acc_ad * mol.mass_a[:,None]
            fbr = np.abs(mol.coeff_s[s])**2 * dfbr
            accbr[s] = np.linalg.norm(fbr / mol.mass_a[:,None])
        return accbr

    def update_nuclear(self, mols: list[Molecule], dt: float):
        mol = mols[-1]
        mol.phase += 0.5 * mol.kinetic_energy * dt
        temp = super().update_nuclear(mols, dt)
        mol.phase += 0.5 * mol.kinetic_energy * dt
        return temp

    def adjust_nuclear(self, mols: list[Molecule], dt: float):
        mol = mols[-1]
        accbr = self._calculate_breaking(mol)

        for s in range(mol.n_states):
            nac = np.sqrt(np.sum(mol.nacdt_ss[s]**2))
            print(f"{s} {accbr[s]} {nac}")
            if accbr[s] > self._dclone and nac < self._dnac and mol.nspawn < self._maxspawn:
                mol.split = [s]
                mol.nspawn += 1
                break

    #TODO: move to molecule
    def dat_header(self, traj: Trajectory):
        dic = super().dat_header(traj)
        dic["phs"] = Printer.write("Phase", "s")
        return dic

    def dat_dict(self, traj: Trajectory):
        dic = super().dat_dict(traj)
        dic["phs"] = Printer.write(traj.mol.phase, "f")
        return dic

    def h5_dict(self, traj: Trajectory):
        dic = super().h5_dict(traj)
        dic["phase"] = traj.mol.phase
        return dic