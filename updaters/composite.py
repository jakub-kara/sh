import numpy as np
from .updaters import Updater
from .nuclear import NuclearUpdater
from classes.meta import Singleton

class CompositeIntegrator(Updater, metaclass = Singleton):
    def __init__(self, *, nuc_upd):
        self._state = 0
        self._count = 0
        self._upds: dict[int, NuclearUpdater] = {}

        base = NuclearUpdater[nuc_upd]()
        self._upds[0] = base
        if base.steps > 1:
            self._upds[-1] = NuclearUpdater[f"rkn{base.steps}"]()

        self.to_init()

    @property
    def steps(self):
        return max(upd.steps for upd in self._upds.values())

    @property
    def active(self):
        return self._upds[self._state]

    def to_init(self):
        self._count = 0
        self._state = min(self._upds.keys())

    def _set_state(self):
        if self._state == -1:
            if self._count >= self._upds[0].steps:
                self._state = 0

    def run(self, *args, **kwargs):
        self._set_state()
        self._count += 1
        self.active.run(*args, **kwargs)

    def save(self):
        return {"state": self._state,
                "count": self._count,
                "upds": self._upds}

    def adjust(self, *, state, count, upds):
        self._state = state
        self._count = count
        self._upds = upds