import numpy as np
from classes.molecule import Molecule
from classes.trajectory import TrajectoryDynamics
from electronic.electronic import ESTProgram

class SimpleEhrenfest(TrajectoryDynamics):
    def __init__(self, config: dict):
        super().__init__(config)

        dectypes = {
            "none": self._decoherence_none,
        }

        dyn = config["dynamics"]
        self._name = "simple ehrenfest"

        inistate = dyn["initstate"]
        self._state = inistate

        self._decoherence = self._decoherence_none

    def calculate_acceleration(self, mol: Molecule):
        mol.acc_ad = 0
        print(mol.acc_ad[0,0])
        for s1 in range(mol.n_states):
            mol.acc_ad -= np.abs(mol.pes.coeff_s[s1])**2 * mol.pes.grad_sad[s1]
            for s2 in range(s1):
                mol.acc_ad -= 2*np.real(np.conj(mol.pes.coeff_s[s1]) * mol.pes.coeff_s[s2])* \
                        mol.pes.nacdr_ssad[s1,s2] * (mol.pes.ham_eig_ss[s2,s2] - mol.pes.ham_eig_ss[s1,s1])

        mol.acc_ad /= mol.mass_a[:,None]

    def potential_energy(self, mol: Molecule):
        poten = 0
        for s in range(mol.n_states):
            poten += np.abs(mol.pes.coeff_s[s])**2 * mol.pes.ham_eig_ss[s,s]
        return poten

    def setup_est(self, est: ESTProgram, mode: str):
        est.all_grads().all_nacs()

    def _decoherence_none(self, *args):
        pass

    def adjust_nuclear(self, *args):
        pass

    def prepare_traj(self):
        self.mol.pes.coeff_s[self._state] = 1
        super().prepare_traj()

    def write_outputs(self):
        super().write_outputs()

        print(self._step, self._time)
        print(np.abs(self.mol.pes.coeff_s)**2)