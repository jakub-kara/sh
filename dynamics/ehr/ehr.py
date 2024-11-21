import numpy as np
from classes.molecule import Molecule
from classes.out import Output
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram

class SimpleEhrenfest(Dynamics, key = "ehr"):
    mode = "g"

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)

        acctypes = {
            "nac": self._acc_nac,
            "eff": self._acc_eff,
        }

        inistate = dynamics["initstate"]
        self._state = inistate
        acctype = dynamics.get("force", "nac")
        if acctype == "nac":
            self.mode += "n"
        self.__class__._acc = acctypes[acctype]

    def potential_energy(self, mol: Molecule):
        poten = 0
        for s in range(mol.n_states):
            poten += np.abs(mol.coeff_s[s])**2 * mol.ham_eig_ss[s,s]
        return poten

    def setup_est(self, mode: str):
        est = ESTProgram()
        if "g" in mode:
            est.all_grads()
        if "o" in mode:
            est.add_ovlp()
        if "n" in mode:
            est.all_nacs()

    def _acc(self, mol: Molecule):
        pass

    def _acc_nac(self, mol: Molecule):
        mol.acc_ad[:] = 0
        for s1 in range(mol.n_states):
            mol.acc_ad -= np.abs(mol.coeff_s[s1])**2 * mol.grad_sad[s1]
            for s2 in range(mol.n_states):
                mol.acc_ad -= 2 * np.real(np.conj(mol.coeff_s[s1]) * mol.coeff_s[s2])* \
                        mol.nacdr_ssad[s1,s2] * mol.ham_eig_ss[s2,s2]

        mol.acc_ad /= mol.mass_a[:,None]

    def _acc_eff(self, mol: Molecule):
        force = np.zeros_like(mol.acc_ad)
        nac_eff_pre = self._get_eff_nac(mol)

        for i in range(mol.n_states):
            force -= np.abs(mol.coeff_s[i])**2 * mol.grad_sad[i]
            for j in range(mol.n_states):
                if i == j:
                    continue
                force += 2 * np.real(mol.nacdt_ss[i,j] * mol.coeff_s[j].conj() * mol.coeff_s[i] * nac_eff_pre[i,j] / (np.sum(nac_eff_pre[i,j] * mol.vel_ad))) * mol.ham_eig_ss[i,i]

        mol.acc_ad = force / mol.mass_a[:,None]

    def calculate_acceleration(self, mol: Molecule):
        self._acc(mol)

    def _get_projector(self, mol: Molecule):
        proj = np.zeros((mol.n_atoms, mol.n_atoms, 3, 3))
        inv_iner = np.linalg.inv(mol.inertia)

    def adjust_nuclear(self, mols: list[Molecule]):
        # mol = mols[-1]
        # print(f"Final pops: {np.abs(mol.coeff_s)**2}")
        # print(f"Check sum:  {np.sum(np.abs(mol.coeff_s)**2)}")
        pass

    def prepare_traj(self, mol: Molecule):
        out = Output()
        out.write_log(f"Initial state:      {self._state}")
        mol.coeff_s[self._state] = 1
        out.write_log("\n")
        super().prepare_traj(mol)