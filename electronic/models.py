import numpy as np
from classes.molecule import Molecule
from electronic.electronic import ESTProgram

class Model1D(ESTProgram, key = "model_1d"):
    def __init__(self, **config):
        super().__init__(**config)
        self._geo = None
        self._hamdia = None
        self._hameig = None
        self._nacdr = None
        self._grad = None
        self._gradham = None
        self._trans = None

    def _select_method(self, key):
        methods = {
            "sub_1": self.sub_1,
            "tully_2": self.tully_2,
            "tully_3": self.tully_3,
        }
        return methods[key]


    def write(self, mol: Molecule):
        self._geo = mol.pos_ad
        if self._hamdia is None:
            self._natoms = mol.n_atoms
            self._hamdia = np.zeros((self._nstates, self._nstates))
            self._hameig = np.zeros((self._nstates, self._nstates))
            self._nacdr = np.zeros((self._nstates, self._nstates, self._natoms, 3))
            self._grad = np.zeros((self._nstates, self._natoms, 3))
            self._gradham = np.zeros((self._nstates, self._nstates, self._natoms, 3))
            self._trans = np.zeros((self._nstates, self._nstates))
        self._method()

    def execute(self):
        pass

    def _diagonalise(self):
        eval, evec = np.linalg.eigh(self._hamdia)
        self._trans = evec
        self._hameig = np.diag(eval)

    def _get_grad_nac(self):
        self._grad = np.einsum("is, ijad, js -> sad", self._trans.conj(), self._gradham, self._trans)
        temp = np.einsum("is, ijad, jr -> srad", self._trans.conj(), self._gradham, self._trans)
        self._nacdr = temp * (1 - np.eye(self._nstates)[:, :, None, None])
        for s1 in range(self._nstates):
            for s2 in range(self._nstates):
                if s1 == s2: continue
                self._nacdr[s1,s2] /= self._hameig[s2,s2] - self._hameig[s1,s1]

    def read_ham(self):
        return self._hameig.copy()

    def read_grad(self):
        return self._grad.copy()

    def read_nac(self):
        return self._nacdr.copy()

    def read_ovlp(self):
        raise NotImplementedError

    def sub_1(self):
        a = 0.01
        b = 0.6
        c = 0.001
        d = 1

        x = self._geo[0,0]
        self._hamdia[:] = 0
        self._hameig[:] = 0
        self._grad[:] = 0
        self._nacdr[:] = 0
        self._gradham[:] = 0

        self._hamdia[0,0] = a * np.tanh(b * x)
        self._hamdia[1,1] = -self._hamdia[0,0]
        self._gradham[0,0,0,0] = a * b * (1 - np.tanh(b * x)**2)
        self._gradham[1,1,0,0] = -a * b * (1 - np.tanh(b * x)**2)

        self._hamdia[0,1] = c * np.exp(-d * x**2)
        self._hamdia[1,0] = self._hamdia[0,1]
        self._gradham[0,1,0,0] = -2 * c * d * x * np.exp(-d * x**2)
        self._gradham[1,0,0,0] = self._gradham[0,1,0,0]

        self._diagonalise()
        self._get_grad_nac()

    def tully_2(self):
        x = self._geo[0,0]
        self._hamdia[:] = 0
        self._hameig[:] = 0
        self._grad[:] = 0
        self._nacdr[:] = 0
        self._gradham[:] = 0

        a = 0.1
        b = 0.28
        c = 0.015
        d = 0.06
        e = 0.05

        self._hamdia[0,0] = 0.
        self._hamdia[1,1] = -a * np.exp(-b * x**2) + e
        self._hamdia[0,1] = c * np.exp(-d * x**2)
        self._hamdia[1,0] = self._hamdia[0,1]

        self._gradham[1,1,0,0] = 2 * a * b * x * np.exp(-b * x**2)
        self._gradham[0,1,0,0] = -2 * c * d * x * np.exp(-d * x**2)
        self._gradham[1,0,0,0] = self._gradham[0,1,0,0]

        self._diagonalise()
        self._get_grad_nac()


    def tully_3(self):
        x = self._geo[0,0]
        self._hamdia[:] = 0
        self._hameig[:] = 0
        self._grad[:] = 0
        self._nacdr[:] = 0
        self._gradham[:] = 0

        a = 6e-4
        b = 0.1
        c = 0.9

        self._hamdia[0,0] = a
        self._hamdia[1,1] = -a
        if x < 0:
            self._hamdia[0,1] = b * np.exp(c * x)
        else:
            self._hamdia[0,1] = b * (2 - np.exp(-c * x))
        self._hamdia[1,0] = self._hamdia[0,1]

        if x < 0:
            self._gradham[0,1,0,0] = b * c * np.exp(c * x)
        else:
            self._gradham[0,1,0,0] = b * c * np.exp(-c * x)
        self._gradham[1,0,0,0] = self._gradham[0,1,0,0]

        self._diagonalise()
        self._get_grad_nac()
