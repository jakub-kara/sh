import numpy as np
from copy import deepcopy
from .ehr import SimpleEhrenfest
from classes.molecule import Molecule

class MultiEhrenfest(SimpleEhrenfest, key = "mce"):
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
        self._phase += 0.5 * mols[-1].kinetic_energy * self.dt
        temp = super().update_nuclear(mols, dt)
        self._phase += 0.5 * mols[-1].kinetic_energy * self.dt
        return temp

    def adjust_nuclear(self, mols: list[Molecule], dt: float):
        mol = mols[-1]
        accbr = self._calculate_breaking(mol)
        # print(self._accbr)
        # print(np.abs(self.mol.nacdt_ss[0,1]))

        # TODO: consider all subsets of states
        # https://doi.org/10.1021/acs.jctc.1c00131
        coeff = mol.coeff_s
        mx = np.argmax(np.abs(coeff))
        # w = 1/np.sum(np.abs(coeff)**4)
        # cond1 = w > 1.3

        # fmean = -np.einsum("s,sad->ad", np.abs(coeff)**2, self.mol.grad_sad)
        fmean = mol.acc_ad * mol.mass_a[:,None]
        fmax = -mol.grad_sad[mx]
        theta = np.arccos((2 * np.sum(fmean * fmax)) / (np.sum(fmean**2) + np.sum(fmax**2)))
        cond2 = theta > np.pi/12

        delta = np.sum(np.abs(2 * np.real(coeff/coeff[mx]) * mol.nacdt_ss[:,mx]))
        cond3 = delta < 5e-3

        print(theta, delta)

        if cond2 and cond3 and self._nspawn < self._maxspawn:
            self.split = [mx]
            self._nspawn += 1

    # TODO: h5_info with widths

    # def h5_dict(self):
    #     dic = super().h5_dict()
    #     dic["phase"] = self._phase
    #     return dic