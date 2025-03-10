import numpy as np
from classes.molecule import Molecule
from classes.out import Output
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram
from updaters.coeff import CoeffUpdater
from updaters.tdc import TDCUpdater

class SimpleEhrenfest(Dynamics):
    key = "ehr"

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)

        nactypes = {
            "nac": self._nac,
            "eff": self._eff_nac,
        }

        self._nactype = dynamics.get("force", "nac")
        self._force_tensor = nactypes[self._nactype]

    def mode(self, mol):
        temp = ["g", CoeffUpdater().mode, TDCUpdater().mode]
        if self._nactype == "nac":
            temp.append("n")
        return temp

    def _nac(self, mol: Molecule):
        return mol.nacdr_ssad

    def read_coeff(self, mol, file = None):
        if file is None:
            mol.coeff_s[mol._state] = 1.
            return
        super().read_coeff(mol, file)

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
                force += 2 * np.real(mol.nacdt_ss[i,j] * mol.coeff_s[j].conj() * mol.coeff_s[i] * nac[i,j] / (np.sum(nac[i,j] * mol.vel_ad))) * mol.ham_eig_ss[i,i]
        mol.acc_ad = force / mol.mass_a[:,None]