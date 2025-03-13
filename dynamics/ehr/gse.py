import numpy as np
from .ehr import SimpleEhrenfest
from classes.molecule import Molecule
from electronic.base import ESTProgram

class GSE(SimpleEhrenfest):
    key = "gse"

    def adjust_nuclear(self, mols: list[Molecule], dt: float):
        mol = mols[-1]
        print(f"Final pops: {np.abs(mol.coeff_s)**2}")
        print(f"Check sum:  {np.sum(np.abs(mol.coeff_s)**2)}")
        print(f"Total en:   {self.potential_energy(mol) + mol.kinetic_energy}")

    def _get_eff_nac(self, mol: Molecule):
        nac_eff_pre = np.zeros_like(mol.nacdr_ssad)
        for i in range(mol.n_states):
            for j in range(i):
                diff = mol.grad_sad[i] - mol.grad_sad[j]
                alpha = (mol.nacdt_ss[i,j] - np.sum(diff * mol.vel_ad)) / np.sum(mol.vel_ad**2)
                nac_eff_pre[i,j] = diff + alpha * mol.vel_ad
                nac_eff_pre[j,i] = -nac_eff_pre[i,j]
        return nac_eff_pre

    def calculate_acceleration(self, mol: Molecule):
        force = np.zeros_like(mol.acc_ad)
        nac_eff_pre = self._get_eff_nac(mol)

        for i in range(mol.n_states):
            force -= np.abs(mol.coeff_s[i])**2 * mol.grad_sad[i]
            for j in range(mol.n_states):
                if i == j:
                    continue
                force += 2 * np.real(mol.nacdt_ss[i,j] * mol.coeff_s[j].conj() * mol.coeff_s[i] * nac_eff_pre[i,j] / (np.sum(nac_eff_pre[i,j] * mol.vel_ad))) * mol.ham_eig_ss[i,i]

        mol.acc_ad = force / mol.mass_a[:,None]