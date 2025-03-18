import numpy as np
from classes.molecule import Molecule
from dynamics.base import Dynamics

class SimpleEhrenfest(Dynamics):
    key = "ehr"

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)
        config["nuclear"]["mixins"] = "ehr"

        nactypes = {
            "nac": self._nac,
            "eff": self._eff_nac,
        }

        self._nactype = dynamics.get("force", "nac")
        self._force_tensor = nactypes[self._nactype]

    def mode(self, mol):
        temp = super().mode(mol)
        temp.append("g")
        if self._nactype == "nac":
            temp.append("n")
        return temp

    def _nac(self, mol: Molecule):
        return mol.nacdr_ssad

    def potential_energy(self, mol: Molecule):
        poten = 0
        for s in range(mol.n_states):
            poten += np.abs(mol.coeff_s[s])**2 * mol.ham_eig_ss[s,s]
        return poten

    def calculate_acceleration(self, mol: Molecule):
        force = np.zeros_like(mol.acc_ad)
        nac = self._force_tensor(mol)

        for i in range(mol.n_states):
            force -= np.abs(mol.coeff_s[i])**2 * mol.grad_sad[i]
            for j in range(mol.n_states):
                if i == j:
                    continue
                if np.linalg.norm(nac[i,j]) > 1e-12:
                    force += 2 * np.real(mol.nacdt_ss[i,j] * mol.coeff_s[i].conj() * mol.coeff_s[j] * nac[i,j] / (np.sum(nac[i,j] * mol.vel_ad))) * mol.ham_eig_ss[i,i]
                else:
                    force += 2 * np.real(mol.nacdt_ss[i,j] * mol.coeff_s[i].conj() * mol.coeff_s[j] * mol.vel_ad / np.sum(mol.vel_ad**2)) * mol.ham_eig_ss[i,i]
        mol.acc_ad = force / mol.mass_a[:,None]