import numpy as np
from classes.molecule import Molecule
from classes.out import Output
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram

class SimpleEhrenfest(Dynamics, key = "ehr"):
    mode = "g"

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)

        nactypes = {
            "nac": self._nac,
            "eff": self._eff_nac,
        }

        inistate = dynamics["initstate"]
        self._state = inistate
        nactype = dynamics.get("force", "nac")
        if nactype == "nac":
            self.mode += "n"
        self._force_tensor = nactypes[nactype]

    def _nac(self, mol: Molecule):
        return mol.nacdr_ssad

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