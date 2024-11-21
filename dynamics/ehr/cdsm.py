import numpy as np
from scipy.linalg import expm
from .ehr import SimpleEhrenfest
from dynamics.sh.checker import HoppingUpdater
from classes.molecule import MoleculeCSDM
from updaters.coeff import CoeffUpdater

class CSDM(SimpleEhrenfest, key = "csdm"):
    def __init__(self, *, dynamics, **config):
        config["nuclear"]["pes"] = "csdm"
        super().__init__(dynamics=dynamics, **config)
        HoppingUpdater(key = dynamics["prob"], **config["quantum"])

        self._pointer = self._state
        self._coeff_co = None

    def prepare_traj(self, mol: MoleculeCSDM):
        super().prepare_traj(mol)
        mol.coeff_co_s[:] = mol.coeff_s

    def dec_vec(self, mol: MoleculeCSDM):
        vec = np.zeros_like(mol.grad_sad)
        for i in range(mol.n_states):
            if i == self._pointer:
                continue
            nac = mol.nacdr_ssad[i, self._pointer]
            vec[i] = np.real(np.sum(mol.mom_ad * nac) / np.linalg.norm(nac) * nac) + mol.mom_ad
        return vec

    def decay_time(self, mol: MoleculeCSDM, vec: np.ndarray = None):
        C = 1
        E0 = 0.1

        if vec is None:
            vec = self.dec_vec(mol)
        norm = np.zeros_like(vec)
        for i in range(mol.n_states):
            if i == self._pointer:
                continue
            norm[i] = vec[i] / np.linalg.norm(vec[i], axis=1)[:,None]
        tau = np.zeros(mol.n_states)

        for i in range(mol.n_states):
            if i == self._pointer:
                continue
            tau[i] = 1 / np.abs(mol.ham_eig_ss[i,i] - mol.ham_eig_ss[self._pointer, self._pointer])
            tau[i] *= C + 4 * E0 / np.sum(mol.mass_a * np.einsum("ad, ad -> a", mol.vel_ad, norm[i])**2)
        tau /= 100
        print(f"pointer: {self._pointer}")
        print(f"tau: {tau}")
        return tau

    def update_quantum(self, mols: list[MoleculeCSDM]):
        self.update_tdc(mols)
        self.update_coeff(mols)
        self.update_pointer(mols)
        self._decoherence_csdm(mols[-1])

    def update_coeff(self, mols: list[MoleculeCSDM]):
        cupd = CoeffUpdater()
        cupd.elapsed(self.curr_step)
        cupd.run(mols, self.dt)
        mols[-1].coeff_s[:] = cupd.coeff.out

        self._swap_coeffs(mols)
        cupd.run(mols, self.dt)
        mols[-1].coeff_co_s[:] = cupd.coeff.out
        self._swap_coeffs(mols)

    def update_pointer(self, mols: list[MoleculeCSDM]):
        hop = HoppingUpdater()
        self._swap_coeffs(mols)
        hop.elapsed(self._step)
        hop.run(mols, self._dt, self._pointer)
        self._pointer = hop.hop.out
        self._swap_coeffs(mols)

    def _decoherence_csdm(self, mol: MoleculeCSDM):
        decay = self.decay_time(mol)
        tot = 0
        for i in range(mol.n_states):
            if i == self._pointer:
                continue
            else:
                print(f"before: {np.abs(mol.coeff_s[i])**2}")
                mol.coeff_s[i] *= np.exp(-1 / (2 * decay[i]) * self.dt)
                print(f"after:  {np.abs(mol.coeff_s[i])**2}")
                print(f"dec {i}: {np.exp(-1 / (2 * decay[i]) * self.dt)}")
                tot += np.abs(mol.coeff_s[i])**2

        mol.coeff_s[self._pointer] *= np.sqrt((1 - tot) / np.abs(mol.coeff_s[self._pointer])**2)

    def _swap_coeffs(self, mols: list[MoleculeCSDM]):
        print(mols[-1].coeff_s)
        print(mols[-1].coeff_co_s)
        for mol in mols:
            temp = mol.coeff_s.copy()
            mol.coeff_s[:] = mol.coeff_co_s
            mol.coeff_co_s[:] = temp
        print(mols[-1].coeff_s)
        print(mols[-1].coeff_co_s)

    def _check_min(self, mols: list[MoleculeCSDM]):
        def nac_sum(mol: MoleculeCSDM):
            tot = 0
            for i in range(mol.n_states):
                tot += np.linalg.norm(mol.nacdr_ssad[i, self._pointer])
            return tot

        temp = nac_sum(mols[-2])
        return (nac_sum(mols[-3]) > temp and nac_sum(mols[-1]) > temp)

    def _reset_coeff(self, mol: MoleculeCSDM):
        mol.coeff_co_s[:] = mol.coeff_s

    def calculate_acceleration(self, mol: MoleculeCSDM):
        super().calculate_acceleration(mol)

        fde = np.zeros_like(mol.acc_ad)
        vec = self.dec_vec(mol)
        tau = self.decay_time(mol, vec)
        dmat = mol.dmat_ss
        for i in range(mol.n_states):
            if i == self._pointer:
                continue
            fde += dmat[i,i] / tau[i, self._pointer] * (mol.ham_eig_ss[i,i] - mol.ham_eig_ss[self._pointer, self._pointer]) / np.sum(vec[i, self._pointer] * mol.vel_ad)
        mol.acc_ad += fde / mol.mass_a[:, None]

    def adjust_nuclear(self, mols: list[MoleculeCSDM]):
        if self._check_min(mols):
            self._reset_coeff(mols[-1])
            print("-"*18 + "RESET" + "-"*17)
        print(f"Check sum:        {np.sum(np.abs(mols[-1].coeff_s)**2)}")
        print(f"Pop:    {np.abs(mols[-1].coeff_s)**2}")
        print(f"Check sum co:     {np.sum(np.abs(mols[-1].coeff_co_s)**2)}")
        print(f"Pop co: {np.abs(mols[-1].coeff_co_s)**2}")

        super().adjust_nuclear(mols)