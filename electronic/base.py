import numpy as np
import os
from abc import abstractmethod
from classes.meta import SingletonFactory
from classes.molecule import Molecule
from classes.constants import convert

def est_method(func):
    func._estmethod = True
    return func

class ESTProgram(metaclass = SingletonFactory):
    _methods = {}

    def __init__(self, *, states: list, program: str, method, path: str = "", options: dict = None, refen = 0, **config):
        self._register_methods()

        self._path = path
        if isinstance(states, int):
            self._states = np.array([states])
        else:
            self._states = np.trim_zeros(np.array(states), "b")
        self._nstates = np.sum(self._states)
        self._natoms = None
        self._spinsum = np.cumsum(self._states) - self._states
        self._refen = convert(refen, "au")

        self._method_name = method
        self._method = self._select_method(method)
        if options is None:
            options = {}
        self._options = options
        self._file = program

        self._calc_grad = np.zeros(self._nstates)
        self._calc_nac = np.zeros((self._nstates, self._nstates))
        self._calc_ovlp = False
        self._calc_dip = False

    @property
    def n_states(self):
        return self._nstates

    def _register_methods(self):
        cls = self.__class__
        for key, val in cls.__dict__.items():
            if hasattr(val, "_estmethod"):
                cls._methods[key] = val

    def _select_method(self, key: str):
        return self._methods[key]

    def reset_calc(self):
        self._calc_grad = np.zeros(self._nstates)
        self._calc_nac = np.zeros((self._nstates, self._nstates))
        self._calc_ovlp = False
        return self

    def request(self, mol: Molecule, *args):
        def to_idx(inp: str):
            if inp == "x":
                return None
            elif inp.isdigit():
                return int(inp)

        for arg in args:
            arg += (3 - len(arg)) * "x"
            if arg[0] == "o":
                self._calc_ovlp = True
            elif arg[0] == "g":
                idx = to_idx(arg[1])
                self._calc_grad[idx] = True
            elif arg[0] == "a":
                self._calc_grad[mol.active] = True
            elif arg[0] == "n":
                idx = to_idx(arg[1]), to_idx(arg[2])
                self._calc_nac[idx] = True
                self._calc_nac[idx[::-1]] = True
            elif arg[0] == "d":
                self._calc_dip = True

    def any_grads(self):
        return np.any(self._calc_grad)

    def any_nacs(self):
        return np.any(self._calc_nac)

    def any_ovlp(self):
        return self._calc_ovlp

    def any_dip(self):
        return self._calc_dip

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
                raise RuntimeError("Cannot read nacmes without a reference Molecule.")
            mol.nacdr_ssad = self.read_nac()
            mol.adjust_nacs(ref)

        if self.any_grads():
            mol.grad_sad = self.read_grad()

        if self.any_ovlp():
            if ref is None:
                raise RuntimeError("Cannot read overlaps without a reference Molecule.")
            mol.ovlp_ss = self.read_ovlp(mol.name_a.astype("<U2"), mol.pos_ad, ref.pos_ad)
            mol.adjust_ovlp()

        if self.any_dip():
            mol.dipmom_ssd = self.read_dipmom()

        mol.transform(False)
        os.chdir("..")

    def write(self, mol: Molecule):
        self._natoms = mol.n_atoms
        with open(f"{self._file}.xyz", "w") as file:
            file.write(mol.to_xyz())
        self._method(self)

    def backup_wf(self):
        pass

    def recover_wf(self):
        pass

    def initiate(self): pass

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

    @abstractmethod
    def read_dipmom(self): pass