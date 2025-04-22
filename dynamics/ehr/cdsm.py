import numpy as np
from scipy.linalg import expm
from .ehr import SimpleEhrenfest
from dynamics.sh.checker import HoppingUpdater
from classes.molecule import Molecule
from updaters.coeff import CoeffUpdater

class CSDM(SimpleEhrenfest):
    key = "csdm"

    def __init__(self, *, dynamics, **config):
        super().__init__(dynamics=dynamics, **config)
        config["nuclear"]["mixins"] = "csdm"
        config["nuclear"]["keep"] = 3
        HoppingUpdater[dynamics.get("prob", "tdc")](**dynamics, **config["quantum"])

    def prepare_dynamics(self, mols: list[Molecule], dt: float):
        mol = mols[-1]
        super().prepare_dynamics(mols, dt)
        mol.coeff_co_s[:] = mol.coeff_s

    def dec_vec(self, mol: Molecule):
        vec = np.zeros_like(mol.grad_sad)
        nac = self._force_tensor(mol)
        for i in range(mol.n_states):
            if i == mol.pointer:
                continue
            vec[i] = mol.mom_ad
            temp = nac[i, mol.pointer]
            if np.linalg.norm(temp) > 1e-14:
                vec[i] += np.real(np.sum(mol.mom_ad * temp) / np.linalg.norm(temp) * temp)
        return vec

    def decay_time(self, mol: Molecule):
        C = 1
        E0 = 0.1

        vec = self.dec_vec(mol)
        norm = np.zeros_like(vec)
        for i in range(mol.n_states):
            if i == mol.pointer:
                continue
            norm[i] = vec[i] / np.linalg.norm(vec[i], axis=1)[:,None]
        tau = np.zeros(mol.n_states)

        for i in range(mol.n_states):
            if i == mol.pointer:
                continue
            tau[i] = 1 / np.abs(mol.ham_eig_ss[i,i] - mol.ham_eig_ss[mol.pointer, mol.pointer])
            tau[i] *= C + 4 * E0 / np.sum(mol.mass_a * np.einsum("ad, ad -> a", mol.vel_ad, norm[i])**2)
            # tau[i] *= C + 2 * E0 / mol.kinetic_energy()

        print(f"pointer: {mol.pointer}")
        print(f"tau: {tau}")
        return tau

    def steps_elapsed(self, steps):
        super().steps_elapsed(steps)
        HoppingUpdater().elapsed(steps)

    def update_coeff(self, mols: list[Molecule], dt: float):
        cupd = CoeffUpdater()
        cupd.run(mols, dt)
        mols[-1].coeff_s[:] = cupd.coeff.out

        self._swap_coeffs(mols)
        cupd.run(mols, dt)
        self._swap_coeffs(mols)
        mols[-1].coeff_co_s[:] = cupd.coeff.out

    def update_pointer(self, mols: list[Molecule], dt: float):
        hop = HoppingUpdater()
        mol = mols[-1]
        self._swap_coeffs(mols)
        hop.run(mols, dt)
        mol.pointer = hop.hop.out
        self._swap_coeffs(mols)

    def _decoherence_csdm(self, mol: Molecule, dt: float):
        decay = self.decay_time(mol)
        tot = 0
        for i in range(mol.n_states):
            if i == mol.pointer:
                continue
            else:
                print(f"before: {np.abs(mol.coeff_s[i])**2}")
                mol.coeff_s[i] *= np.exp(-1 / (2 * decay[i]) * dt)
                print(f"after:  {np.abs(mol.coeff_s[i])**2}")
                print(f"dec {i}: {np.exp(-1 / (2 * decay[i]) * dt)}")
                tot += np.abs(mol.coeff_s[i])**2

        mol.coeff_s[mol.pointer] *= np.sqrt((1 - tot) / np.abs(mol.coeff_s[mol.pointer])**2)

    def _swap_coeffs(self, mols: list[Molecule]):
        for mol in mols:
            temp = mol.coeff_s.copy()
            mol.coeff_s[:] = mol.coeff_co_s
            mol.coeff_co_s[:] = temp

    def _check_min(self, mols: list[Molecule]):
        def nac_sum(mol: Molecule):
            return np.sum(np.abs(mol.nacdt_ss[mol.pointer,:]))

        temp = nac_sum(mols[-2])
        return (nac_sum(mols[-3]) > temp and nac_sum(mols[-1]) > temp)

    def _reset_coeff(self, mol: Molecule):
        mol.coeff_co_s[:] = mol.coeff_s

    def calculate_acceleration(self, mol: Molecule):
        super().calculate_acceleration(mol)

        fde = np.zeros_like(mol.acc_ad)
        vec = self.dec_vec(mol)
        tau = self.decay_time(mol)
        for i in range(mol.n_states):
            if i == mol.pointer:
                continue
            fde += np.abs(mol.coeff_s[i])**2 / tau[i] * (mol.ham_eig_ss[i,i] - mol.ham_eig_ss[mol.pointer, mol.pointer]) / np.sum(vec[i] * mol.vel_ad) * vec[i]
        mol.acc_ad += fde / mol.mass_a[:, None]

    def adjust_nuclear(self, mols: list[Molecule], dt: float):
        mol = mols[-1]
        self.update_pointer(mols, dt)
        self._decoherence_csdm(mol, dt)

        if self._check_min(mols):
            self._reset_coeff(mol)
            print("-"*18 + "RESET" + "-"*17)
        print(f"Check sum:        {np.sum(np.abs(mol.coeff_s)**2)}")
        print(f"Pop:    {np.abs(mol.coeff_s)**2}")
        print(f"Check sum co:     {np.sum(np.abs(mol.coeff_co_s)**2)}")
        print(f"Pop co: {np.abs(mol.coeff_co_s)**2}")

        super().adjust_nuclear(mols, dt)