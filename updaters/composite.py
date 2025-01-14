import numpy as np
from .updaters import Updater
from .nuclear import NuclearUpdater
from classes.meta import Singleton

class CompositeIntegrator(Updater, metaclass = Singleton):
    def __init__(self, *, nuc_upd):
        self._state = 0
        self._count = 0
        self._upds: dict[int, NuclearUpdater] = {}

        base = NuclearUpdater(key = nuc_upd)
        self._upds[0] = base
        if base.steps > 1:
            self._upds[-1] = NuclearUpdater(key = "rkn4")

        self.steps = max([i.steps for i in self._upds.values()])

        self.to_init()

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

    def update(self, *args, **kwargs):
        self._set_state()
        self._count += 1
        return self.active.update(*args, **kwargs)