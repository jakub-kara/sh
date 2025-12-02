import numpy as np
from copy import deepcopy
from abc import abstractmethod
from classes.config import Config
from classes.molecule import Molecule
from classes.timestep import Timestep

class Updater:
    name = ""
    steps = 0
    substeps = 1

    def __init__(self, **kwargs):
        self.result = None
        self.success = True

    def is_ready(self, step: int):
        return step >= self.steps - 1

    def run(self, mols: list[Molecule], ts: Timestep, *args, **kwargs):
        self.new_result(mols[-1], *args, **kwargs)
        if self.is_ready(ts.step):
            self.update(mols, ts, *args, **kwargs)
        else:
            self.no_update(mols, ts, *args, **kwargs)
        self.validate(mols, ts, *args, **kwargs)

    @abstractmethod
    def new_result(self, mol: Molecule):
        raise NotImplementedError

    @abstractmethod
    def update(self, mols: list[Molecule], ts: Timestep, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def no_update(self, mols: list[Molecule], ts: Timestep, *args, **kwargs):
        raise NotImplementedError

    def validate(self, mols: list[Molecule], ts: Timestep, *args, **kwargs):
        self.success = True

# Multistage mixin
@Config("n_substeps", default=50, path=Config.QUANTUM, valid=Config.POSITIVE(Config.INT))
class Multistage:
    def __init__(self, *, n_substeps = 50, **kwargs):
        super().__init__(**kwargs)
        self.substeps = n_substeps

class UpdateResult:
    def __init__(self, integrand, n_substeps):
        self.substeps = n_substeps
        self._idx = 0
        self._npa = isinstance(integrand, np.ndarray)
        if self._npa:
            dtype = integrand.dtype
            self._vals = np.zeros((self.substeps + 1, *integrand.shape), dtype=dtype)
            self.push(integrand.copy())
        else:
            dtype = type(integrand)
            self._vals = np.empty(self.substeps + 1, dtype=dtype)
            self.push(deepcopy(integrand))

    @property
    def inp(self):
        return self._copy_val(self._vals[0])

    @property
    def out(self):
        return self._copy_val(self._vals[-1])

    @out.setter
    def out(self, value):
        self._vals[-1] = value

    def _copy_val(self, val):
        if self._npa:
            return val.copy()
        else:
            return deepcopy(val)

    def copy(self):
        return deepcopy(self)

    def push(self, val):
        self._vals[self._idx] = val
        self._idx += 1

    def get(self, idx):
        return self._copy_val(self._vals[idx])

    def fill(self):
        self._vals[:] = self.inp
        self._idx = self.substeps

    def interpolate(self, frac: float):
        return (1 - frac) * self.inp + frac * self.out