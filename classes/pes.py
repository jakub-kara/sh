import numpy as np
from classes.errors import ArrayShapeError

class PES:
    def __init__(self, n_states: int, n_atoms: int, n_dim: int = 3):
        self._refen = None
        self._ham_eig = np.zeros((n_states, n_states))
        self._ham_dia = np.zeros((n_states, n_states), dtype=np.complex128)
        self._trans = np.eye(n_states, dtype=np.complex128)
        self._grad = np.zeros((n_states, n_atoms, n_dim))
        self._nacdr = np.zeros((n_states, n_states, n_atoms, n_dim))
        self._nacdt = np.zeros((n_states, n_states))
        self._ovlp = np.zeros((n_states, n_states))
        self._dipmom = np.zeros((n_states, n_states, n_dim))
        self._nacflp = np.zeros((n_states, n_states))
        self._phase = np.ones(n_states)
        self._coeff = np.zeros(n_states, dtype=np.complex128)

    def adjust_nacs(self, other):
        for s1 in range(self.n_states):
            for s2 in range(s1+1, self.n_states):
                # calculate overlap
                if np.sum(other.nacdr_ssad[s1,s2] * self.nacdr_ssad[s1,s2]) < 0:
                    # flip sign if overlap < 0 and set the flag
                    self.nacdr_ssad[s1,s2] = -self.nacdr_ssad[s1,s2]
                    self._nacflp[s1,s2] = True
                    self._nacflp[s2,s1] = True
                else:
                    self._nacflp[s1,s2] = False
                    self._nacflp[s2,s1] = False
                # nacmes antisymmetric
                self.nacdr_ssad[s2,s1] = -self.nacdr_ssad[s1,s2]

    def adjust_energy(self):
        # set ground-state energy as 0 at first step
        if self._refen is None:
            # self._refen = self.ham_dia_ss[0,0]
            self._refen = 0.
        # adjust potential energy to be consistent
        for s in range(self.n_states):
            self.ham_dia_ss[s,s] -= self._refen

    def diagonalise_ham(self):
        eval, evec = np.linalg.eigh(self.ham_dia_ss)
        self.trans_ss = evec
        self.ham_eig_ss = np.diag(eval)

    def transform(self, diagonalise: bool = False):
        if not diagonalise:
            self.ham_eig_ss[:] = np.real(self.ham_dia_ss)
            self.trans_ss[:] = np.eye(self.n_states)

        # need to transform gradient for non-diagonal hamiltonian
        # for details, see https://doi.org/10.1002/qua.2489
        self.diagonalise_ham()
        g_diab = np.zeros_like(self._nacdr, dtype=np.complex128)
        for i in range(self.n_states):
            for j in range(self.n_states):
                # on-diagonal part
                g_diab[i,j] = (i == j) * self.grad_sad[i]
                # off-diagonal part
                g_diab[i,j] -= (self.ham_dia_ss[i,i] - self.ham_dia_ss[j,j]) * self.nacdr_ssad[i,j]
        # just a big matrix multiplication with some extra dimensions
        g_diag = np.einsum("ij,jkad,kl->ilad", self.trans_ss.conj().T, g_diab, self.trans_ss)

        # only keep the real part of the gradient
        for i in range(self.n_states):
            self.grad_sad[i] = np.real(g_diag[i,i])

    @property
    def n_states(self):
        return self.coeff_s.shape[0]

    @property
    def ham_eig_ss(self):
        return self._ham_eig

    @ham_eig_ss.setter
    def ham_eig_ss(self, value: np.ndarray):
        self._ham_eig[:] = value

    @property
    def ham_dia_ss(self):
        return self._ham_dia

    @ham_dia_ss.setter
    def ham_dia_ss(self, value: np.ndarray):
        self._ham_dia[:] = value

    @property
    def trans_ss(self):
        return self._trans

    @trans_ss.setter
    def trans_ss(self, value: np.ndarray):
        self._trans[:] = value

    @property
    def grad_sad(self):
        return self._grad

    @grad_sad.setter
    def grad_sad(self, value: np.ndarray):
        self._grad[:] = value

    @property
    def nacdr_ssad(self):
        return self._nacdr

    @nacdr_ssad.setter
    def nacdr_ssad(self, value: np.ndarray):
        self._nacdr[:] = value

    @property
    def nacdt_ss(self):
        return self._nacdt

    @nacdt_ss.setter
    def nacdt_ss(self, value: np.ndarray):
        self._nacdt[:] = value

    @property
    def ovlp_ss(self):
        return self._ovlp

    @ovlp_ss.setter
    def ovlp_ss(self, value: np.ndarray):
        self._ovlp[:] = value

    @property
    def dipmom_ssd(self):
        return self._dipmom

    @dipmom_ssd.setter
    def dipmom_ssd(self, value: np.ndarray):
        self._dipmom[:] = value

    @property
    def nac_flip_ss(self):
        return self._nacflp

    @nac_flip_ss.setter
    def nac_flip_ss(self, value: np.ndarray):
        self._nacflp[:] = value

    @property
    def phase_s(self):
        return self._phase

    @phase_s.setter
    def phase_s(self, value: np.ndarray):
        self._phase[:] = value

    @property
    def coeff_s(self):
        return self._coeff

    @coeff_s.setter
    def coeff_s(self, value: np.ndarray):
        self._coeff[:] = value