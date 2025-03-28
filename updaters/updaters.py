import numpy as np
from copy import deepcopy
from abc import abstractmethod
from classes.molecule import Molecule

class Updater:
    name = ""
    steps = 0
    substeps = 1

    def __init__(self, **kwargs):
        self._ready = False

    @property
    def single_step(self):
        return self.substeps > 1

    def elapsed(self, step: int):
        self._ready = step >= self.steps

    def is_ready(self):
        return self._ready

    def run(self, mols: list[Molecule], dt: float, *args, **kwargs):
        self.new_result(mols[-1], *args, **kwargs)
        if self._ready:
            self.update(mols, dt, *args, **kwargs)
        else:
            self.no_update(mols, dt, *args, **kwargs)

    @abstractmethod
    def new_result(self, mol: Molecule):
        raise NotImplementedError

    @abstractmethod
    def update(self, mols: list[Molecule], dt: float, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def no_update(self, mols: list[Molecule], dt: float, *args, **kwargs):
        raise NotImplementedError

# Multistage mixin
class Multistage:
    def __init__(self, *, n_substeps = 50, **kwargs):
        super().__init__(**kwargs)
        self.substeps = n_substeps

class UpdateResult:
    def __init__(self, integrand, n_substeps):
        self.substeps = n_substeps
        self._npa = isinstance(integrand, np.ndarray)
        if self._npa:
            dtype = integrand.dtype
            self._inp = integrand.copy()
            self.inter = np.zeros((n_substeps, *integrand.shape), dtype=dtype)
        else:
            dtype = type(integrand)
            self._inp = deepcopy(integrand)
            self.inter = np.empty(n_substeps, dtype=dtype)

    @property
    def inp(self):
        if self._npa:
            return self._inp.copy()
        else:
            return deepcopy(self._inp)

    @property
    def out(self):
        return self.inter[-1]

    @out.setter
    def out(self, value):
        self.inter[-1] = value

    def copy(self):
        return deepcopy(self)

    def fill(self):
        self.inter[:] = self.inp
        self.out = self.inp

    # janky, maybe rework
    def interpolate(self, frac: float):
        temp = frac * self.substeps
        idx = int(temp)
        if idx >= self.substeps - 1:
            return self.out
        upper = (temp - idx) * self.inter[idx + 1]
        if idx == 0:
            lower = (1 - temp + idx) * self.inp
        else:
            lower = (1 - temp + idx) * self.inter[idx]
        return lower + upper