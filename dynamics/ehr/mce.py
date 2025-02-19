import numpy as np
from copy import deepcopy
from .ehr import SimpleEhrenfest
from classes.molecule import Molecule

class MultiEhrenfest(SimpleEhrenfest):
    key = "mce"

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)

        self._dclone = dynamics.get("dclone", 5e-6)
        self._dnac = dynamics.get("dnac", 2e-3)
        self._maxspawn = dynamics.get("maxspawn", 3)
        self._nspawn = 0
        self._phase = 0

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
        self._phase += 0.5 * mols[-1].kinetic_energy * dt
        temp = super().update_nuclear(mols, dt)
        self._phase += 0.5 * mols[-1].kinetic_energy * dt
        return temp

    def adjust_nuclear(self, mols: list[Molecule], dt: float):
        mol = mols[-1]
        accbr = self._calculate_breaking(mol)

        for s in range(mol.n_states):
            nac = np.sqrt(np.sum(mol.nacdt_ss[s]**2))
            print(f"{s} {accbr[s]} {nac}")
            if accbr[s] > self._dclone and nac < self._dnac and self._nspawn < self._maxspawn:
                self.split = [s]
                self._nspawn += 1
                break

    def h5_dict(self):
        return {"phase": self._phase}