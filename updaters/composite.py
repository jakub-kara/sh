import numpy as np
from .base import Updater
from .nuclear import NuclearUpdater
from classes.constants import convert
from classes.meta import Singleton
from classes.molecule import Molecule

class CompositeIntegrator(Updater, metaclass = Singleton):
    def __init__(self, *, nuc_upd, **kwargs):
        self._iactive = 0
        self._count = 0
        self._success = False
        self._thresh = convert(kwargs.get("thresh", 1e10), "au")
        self._upds: dict[int, NuclearUpdater] = {}

        base = NuclearUpdater.select(nuc_upd)()
        self._upds[0] = base
        if base.steps > 1:
            self._upds[-1] = NuclearUpdater.select(f"rkn{base.steps}")()

        self.to_init()

    @property
    def steps(self):
        return max(upd.steps for upd in self._upds.values())

    @property
    def success(self):
        return self._success

    @property
    def active(self):
        return self._upds[self._iactive]

    def get_state(self):
        return {"iact": self._iactive,
                "count": self._count,
                "upds": self._upds}

    def set_state(self, *, iact, count, upds):
        self._iactive = iact
        self._count = count
        self._upds = upds

    def to_init(self):
        self._count = 0
        self._iactive = min(self._upds.keys())

    def _set_active(self):
        if self._iactive == -1:
            if self._count >= self._upds[0].steps:
                self._iactive = 0

    def run(self, mols: list[Molecule], dt: float):
        self._set_active()
        self._count += 1
        self.active.run(mols, dt)

    def validate(self, val: float):
        self._success = val < self._thresh