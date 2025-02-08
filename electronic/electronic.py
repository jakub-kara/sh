import numpy as np
import os
from abc import abstractmethod
from classes.meta import SingletonFactory
from classes.molecule import Molecule
from classes.constants import convert

class ESTProgram(metaclass = SingletonFactory):
    def __init__(self, *, states: list, program: str, type: str, path: str = "", options: dict = None, refen = 0, **config):
        self._path = path
        if isinstance(states, int):
            self._states = np.array([states])
        else:
            self._states = np.trim_zeros(np.array(states), "b")
        self._nstates = np.sum(self._states)
        self._natoms = None
        self._spinsum = np.cumsum(self._states) - self._states
        self._refen = convert(refen, "au")

        self._type = type

        self._method = self._select_method(type)
        if options is None:
            options = {}
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
            if len(arg) != 2:
                continue
            self._calc_nac[arg] = 1
            self._calc_nac[arg[::-1]] = 1
        return self

    def remove_nacs(self, *args):
        for arg in args:
            if len(arg) != 1:
                continue
            self._calc_nac[arg] = 0
            self._calc_nac[arg[::-1]] = 0
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

    def run(self, mol: Molecule):
        os.chdir("est")
        self.write(mol)
        self.execute()
        os.chdir("..")

    def read(self, mol: Molecule, ref: Molecule = None):
        os.chdir("est")

        mol.ham_dia_ss = self.read_ham()
        mol.adjust_energy(self._refen)

        if self.any_nacs():
            if ref is None:
                raise ValueError("Cannot read nacmes without reference Molecule.")
            mol.nacdr_ssad = self.read_nac()
            mol.adjust_nacs(ref)

        if self.any_grads():
            mol.grad_sad = self.read_grad()

        if self.any_ovlp():
            if ref is None:
                raise ValueError("Cannot read overlaps without reference Molecule.")
            mol.ovlp_ss = self.read_ovlp(mol.name_a.astype("<U2"), mol.pos_ad, ref.pos_ad)
            mol.adjust_ovlp()

        mol.transform(False)
        os.chdir("..")

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
