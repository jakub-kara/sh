import numpy as np
from abc import ABC, abstractmethod
from classes.molecule import Molecule

class ESTProgram(ABC):
    def __init__(self, *, states: list, program: str, type: str, path: str, options: dict, **config):
        self._path = path
        self._states = np.trim_zeros(np.array(states), "b")
        self._nstates = np.sum(self._states)
        self._natoms = None
        self._spinsum = np.cumsum(self._states) - self._states

        self._method = self._select_method(type)
        self._options = options
        self._file = program
        self._calc_grad = np.zeros(self._nstates)
        self._calc_nac = np.zeros((self._nstates, self._nstates))
        self._calc_ovlp = False

    @property
    def n_states(self):
        return self._nstates

    @abstractmethod
    def _select_method(self, key: str):
        pass

    def reset_calc(self):
        self._calc_grad = np.zeros(self._nstates)
        self._calc_nac = np.zeros((self._nstates, self._nstates))
        self._calc_ovlp = False
        return self

    def all_grads(self):
        self._calc_grad[:] = 1
        return self

    def add_grads(self, *args):
        for arg in args:
            self._calc_grad[arg] = 1
        return self

    def remove_grads(self, *args):
        for arg in args:
            self._calc_grad[arg] = 0
        return self

    def any_grads(self):
        return np.any(self._calc_grad)

    def all_nacs(self):
        self._calc_nac[:] = 1
        return self

    def add_nacs(self, *args):
        for arg in args:
            self._calc_nac[arg] = 1
        return self

    def remove_nacs(self, *args):
        for arg in args:
            self._calc_nac[arg] = 0
        return self

    def any_nacs(self):
        return np.any(self._calc_nac)

    def add_ovlp(self):
        self._calc_ovlp = True
        return self

    def remove_ovlp(self):
        self._calc_ovlp = False
        return self

    def any_ovlp(self):
        return self._calc_ovlp

    def write(self, mol: Molecule):
        self._natoms = mol.n_atoms
        with open(f"{self._file}.xyz", "w") as file:
            file.write(mol.to_xyz())
        self._method()

    def backup_wf(self):
        pass

    def recover_wf(self):
        pass

    @abstractmethod
    def execute(self): pass

    @abstractmethod
    def read_ham(self): pass

    @abstractmethod
    def read_grad(self): pass

    @abstractmethod
    def read_nac(self): pass

    @abstractmethod
    def read_ovlp(self): pass