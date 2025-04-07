import numpy as np
import os
from classes.molecule import Molecule
from electronic.base import ESTProgram, est_method

class Model(ESTProgram):
    key = "model"

    def __init__(self, **config):
        super().__init__(**config)
        self._geo = None
        self._hamdia = None
        self._hameig = None
        self._nacdr = None
        self._grad = None
        self._gradham = None
        self._trans = None

    def write(self, mol: Molecule):
        self._geo = mol.pos_ad
        if self._hamdia is None:
            self._natoms = mol.n_atoms
            self._hamdia = np.zeros_like(mol.ham_eig_ss)
            self._trans = np.zeros_like(mol.ham_eig_ss)
            self._hameig = np.zeros_like(mol.ham_eig_ss)
            self._grad = np.zeros_like(mol.grad_sad)
            self._gradham = np.zeros_like(mol.nacdr_ssad)
            self._nacdr = np.zeros_like(mol.nacdr_ssad)
        self._method(self)

    def execute(self):
        self._diagonalise()
        self._get_grad_nac()

    def backup_wf(self):
        np.save("backup/states.npy", self._trans)

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

    def read_ovlp(self, *args, **kwargs):
        old = np.load("../backup/states.npy")
        new = self._trans

        ovl = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                ovl[i,j] = np.sum(old[:,i] * new[:,j])

        return ovl

    def read_dipmom(self):
        raise NotImplementedError

    @est_method
    def sub_1(self):
        a = 0.01
        b = 0.6
        c = 0.001
        d = 1
        x = self._geo[0,0]

        self._hamdia[0,0] = a * np.tanh(b * x)
        self._hamdia[1,1] = -self._hamdia[0,0]
        self._gradham[0,0,0,0] = a * b * (1 - np.tanh(b * x)**2)
        self._gradham[1,1,0,0] = -a * b * (1 - np.tanh(b * x)**2)

        self._hamdia[0,1] = c * np.exp(-d * x**2)
        self._hamdia[1,0] = self._hamdia[0,1]
        self._gradham[0,1,0,0] = -2 * c * d * x * np.exp(-d * x**2)
        self._gradham[1,0,0,0] = self._gradham[0,1,0,0]

    @est_method
    def tully_2(self):
        a = 0.1
        b = 0.28
        c = 0.015
        d = 0.06
        e = 0.05
        x = self._geo[0,0]

        self._hamdia[0,0] = 0.
        self._hamdia[1,1] = -a * np.exp(-b * x**2) + e
        self._hamdia[0,1] = c * np.exp(-d * x**2)
        self._hamdia[1,0] = self._hamdia[0,1]

        self._gradham[1,1,0,0] = 2 * a * b * x * np.exp(-b * x**2)
        self._gradham[0,1,0,0] = -2 * c * d * x * np.exp(-d * x**2)
        self._gradham[1,0,0,0] = self._gradham[0,1,0,0]

    @est_method
    def tully_3(self):
        a = 6e-4
        b = 0.1
        c = 0.9
        x = self._geo[0,0]

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

    @est_method
    def sub_x(self):
        a = 0.03
        b = 1.6
        c = 0.005
        x = self._geo[0,0]

        self._hamdia[0,0] = a*(np.tanh(b*x) + np.tanh(b*(x + 7)))
        self._gradham[0,0,0,0] = a*b*(1 - np.tanh(b*x)**2 + 1 - np.tanh(b*(x + 7))**2)
        self._hamdia[1,1] = -a*(np.tanh(b*x) + np.tanh(b*(x - 7)))
        self._gradham[1,1,0,0] = -a*b*(1 - np.tanh(b*x)**2 + 1 - np.tanh(b*(x - 7))**2)
        self._hamdia[2,2] = -a*(np.tanh(b*(x + 7)) - np.tanh(b*(x - 7)))
        self._gradham[2,2,0,0] = -a*b*(1 - np.tanh(b*(x + 7))**2 - (1 - np.tanh(b*(x - 7))**2))

        self._hamdia[0,1] = c*np.exp(-x**2)
        self._gradham[0,1,0,0] = -2*c*x*np.exp(-x**2)
        self._hamdia[1,0] = self._hamdia[0,1]
        self._gradham[1,0,0,0] = self._gradham[0,1,0,0]

        self._hamdia[0,2] = c*np.exp(-(x + 7)**2)
        self._gradham[0,2,0,0] = -2*c*(x + 7)*np.exp(-(x + 7)**2)
        self._hamdia[2,0] = self._hamdia[0,2]
        self._gradham[2,0,0,0] = self._gradham[0,2,0,0]

        self._hamdia[1,2] = c*np.exp(-(x - 7)**2)
        self._gradham[1,2,0,0] = -2*c*(x - 7)*np.exp(-(x - 7)**2)
        self._hamdia[2,1] = self._hamdia[1,2]
        self._gradham[2,1,0,0] = self._gradham[1,2,0,0]

    @est_method
    def sub_s(self):
        a = 0.015
        b = 1.0
        c = 0.005
        d = 0.5
        e = 0.1
        x = self._geo[0,0]

        self._hamdia[0,0] = a*(np.tanh(b*(x - 7)) - np.tanh(b*(x + 7))) + a
        self._gradham[0,0,0,0] = a*b*(1 - np.tanh(b*(x - 7))**2 - (1 - np.tanh(b*(x + 7))**2))
        self._hamdia[1,1] = -self._hamdia[0,0]
        self._gradham[1,1,0,0] = -self._gradham[0,0,0,0]
        self._hamdia[2,2] = 2*a*np.tanh(d*x)
        self._gradham[2,2,0,0] = 2*a*d*(1 - np.tanh(d*x)**2)

        self._hamdia[0,1] = c*(np.exp(-e*(x - 7)**2) + np.exp(-e*(x + 7)**2))
        self._gradham[0,1,0,0] = -2*c*e*((x - 7)*np.exp(-e*(x - 7)**2) + (x + 7)*np.exp(-e*(x + 7)**2))
        self._hamdia[1,0] = self._hamdia[0,1]
        self._gradham[1,0,0,0] = self._gradham[0,1,0,0]

        self._hamdia[0,2] = c*np.exp(-e*x**2)
        self._gradham[0,2,0,0] = -2*c*e*x*np.exp(-e*x**2)
        self._hamdia[2,0] = self._hamdia[0,2]
        self._gradham[2,0,0,0] = self._gradham[0,2,0,0]

        self._hamdia[1,2] = c*np.exp(-e*x**2)
        self._gradham[1,2,0,0] = -2*c*e*x*np.exp(-e*x**2)
        self._hamdia[2,1] = self._hamdia[1,2]
        self._gradham[2,1,0,0] = self._gradham[1,2,0,0]

    @est_method
    def ho(self):
        x = self._geo[0,0]
        self._hamdia[0,0] = x**2 / 2
        self._gradham[0,0,0,0] = x

    @est_method
    def ho2(self):
        a = 0.005
        b = 2
        c = 2
        d = 0

        # a = 0.005
        # b = 2
        # c = 2
        # d = 10
        x = self._geo[0,0]

        self._hamdia[0,0] = a * (x - b)**2
        self._gradham[0,0,0,0] = 2 * a * (x - b)
        self._hamdia[1,1] = a * (x + b)**2
        self._gradham[1,1,0,0] = 2 * a * (x + b)

        # self._hamdia[0,1] = c * np.exp(-d * x**2)
        # self._hamdia[1,0] = self._hamdia[0,1]
        # self._gradham[0,1,0,0] = - 2 * c * d * x * np.exp(-d * x**2)
        # self._gradham[1,0,0,0] = self._gradham[0,1,0,0]

        self._hamdia[0,1] = a*c
        self._hamdia[1,0] = a*c
        self._gradham[0,1,0,0] = 0
        self._gradham[1,0,0,0] = 0

    @est_method
    def ho3(self):
        a = 0.005
        b = 4
        c = 0.02
        d = 0.2
        x = self._geo[0,0]

        self._hamdia[0,0] = a * (x - 2*b)**2
        self._hamdia[1,1] = a * x**2
        self._hamdia[2,2] = a * (x + 2*b)**2
        self._gradham[0,0,0,0] = 2 * a * (x - 2*b)
        self._gradham[1,1,0,0] = 2 * a * x
        self._gradham[2,2,0,0] = 2 * a * (x + 2*b)

        # self._hamdia[0,1] = c * np.exp(-d * x**2)
        # self._hamdia[1,0] = self._hamdia[0,1]
        # self._gradham[0,1,0,0] = - 2 * c * d * x * np.exp(-d * x**2)
        # self._gradham[1,0,0,0] = self._gradham[0,1,0,0]

        self._hamdia[0,1] = c
        self._hamdia[1,0] = c
        self._gradham[0,1,0,0] = 0
        self._gradham[1,0,0,0] = 0

        self._hamdia[1,2] = c
        self._hamdia[2,1] = c
        self._gradham[1,2,0,0] = 0
        self._gradham[2,1,0,0] = 0

        self._hamdia[0,2] = c
        self._hamdia[2,0] = c
        self._gradham[0,2,0,0] = 0
        self._gradham[2,0,0,0] = 0

    @est_method
    def hh(self):
        a = 0.005
        b = -1
        d = 0.5
        k = 0.3
        x = self._geo[0,0]
        y = self._geo[0,1]

        f = lambda x, y: a * (1/2 * (x**2 + y**2) + k * (x*y**2 - 1/3*x**3) + 1/16 * k**2 * (x**2 + y**2)**2)
        self._hamdia[0,0] = f(x - b, y)
        self._hamdia[1,1] = f(-x - b, y)
        # self._hamdia[0,1] = a*d*y
        # self._hamdia[1,0] = a*d*y
        self._hamdia[0,1] = a*d
        self._hamdia[1,0] = a*d

        dfdx = lambda x, y: a * (x + k * (y**2 - x**2) + 1/4 * k**2 * (x**3 + x*y**2))
        dfdy = lambda x, y: a * (y + 2*k*x*y + 1/4 * k**2 * (x**2*y + y**3))
        self._gradham[0,0,0,0] = dfdx(x - b, y)
        self._gradham[0,0,0,1] = dfdy(x - b, y)
        self._gradham[1,1,0,0] = -dfdx(-x - b, y)
        self._gradham[1,1,0,1] = dfdy(-x - b, y)
        # self._gradham[0,1,0,1] = a*d
        # self._gradham[1,0,0,1] = a*d

    @est_method
    def ho2_2(self):
        a = 0.005
        b = 2
        c = 2

        x = self._geo[0,0]
        y = self._geo[0,1]

        self._hamdia[0,0] = a * ((x - b)**2 + y**2)
        self._gradham[0,0,0,0] = 2 * a * (x - b)
        self._gradham[0,0,0,1] = 2 * a * y
        self._hamdia[1,1] = a * ((x + b)**2 + y**2)
        self._gradham[1,1,0,0] = 2 * a * (x + b)
        self._gradham[1,1,0,1] = 2 * a * y

        # self._hamdia[0,1] = a * c * y
        self._hamdia[0,1] = a * c
        self._hamdia[1,0] = self._hamdia[0,1]
        # self._gradham[0,1,0,1] = a * c
        # self._gradham[1,0,0,1] = a * c