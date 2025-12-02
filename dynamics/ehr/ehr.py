import numpy as np
from classes.config import Config, get_cfg
from classes.molecule import Molecule, MoleculeMixin
from dynamics.base import Dynamics
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater
from electronic.base import ESTMode

@Config.Option("ehr", "method")
@Config("force", default="nac", valid=Config.STR, path=Config.DYNAMICS)
class SimpleEhrenfest(Dynamics):
    mode = ESTMode(ESTMode.GRADIENTS)
    _molmixins = ["ehr"]

    def __init__(self, **config):
        super().__init__(**config)

        nactypes = {
            "nac": self._nac,
            "eff": self._eff,
        }

        self._nactype = get_cfg(self, "force", config)
        if self._nactype == "nac":
            self.__class__.mode = ESTMode(ESTMode.GRADIENTS + ESTMode.NACMES)
        self._force_tensor = nactypes[self._nactype]

    def _nac(self, mol: Molecule):
        return mol.nacdr_ssad

    def _eff(self, mol: Molecule):
        return mol.eff_nac()

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
                if self._nactype == "nac":
                    force += 2 * np.real(mol.coeff_s[i].conj() * mol.coeff_s[j] * mol.nacdr_ssad[i,j] * mol.ham_eig_ss[i,i])
                elif np.linalg.norm(nac[i,j]) > 1e-12:
                    force += 2 * np.real(mol.nacdt_ss[i,j] * mol.coeff_s[i].conj() * mol.coeff_s[j] * nac[i,j] / (np.sum(nac[i,j] * mol.vel_ad))) * mol.ham_eig_ss[i,i]
                else:
                    force += 2 * np.real(mol.nacdt_ss[i,j] * mol.coeff_s[i].conj() * mol.coeff_s[j] * mol.vel_ad / np.sum(mol.vel_ad**2)) * mol.ham_eig_ss[i,i]
        mol.acc_ad = force / mol.mass_a[:,None]

@Config.Option("ehr", "mixin")
class EhrMixin(MoleculeMixin):
    pass