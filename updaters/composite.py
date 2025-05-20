import numpy as np
from .base import Updater
from .nuclear import NuclearUpdater
from classes.constants import convert
from classes.meta import Singleton
from classes.molecule import Molecule
from classes.out import Output as out

class CompositeIntegrator(Updater, metaclass = Singleton):
    def __init__(self, *, nuc_upd = "vv", **kwargs):
        self._iactive = 0
        self._count = 0
        self._success = False
        self._thresh = convert(kwargs.get("thresh", 1e10), "au")
        self._upds: dict[int, NuclearUpdater] = {}

        if not isinstance(nuc_upd, (list, tuple)):
            nuc_upd = [nuc_upd]

        for i, upd in enumerate(nuc_upd):
            self._upds[i] = NuclearUpdater.select(upd)()

        base = self._upds[0]
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

        temp: Molecule = self.active.out.out
        self.validate(np.abs(temp.total_energy() - mols[-1].total_energy()))

    def validate(self, val):
        self._success = val < self._thresh
        print("DIFF", convert(val, "au", "ev"))
        if self._success:
            self._iactive = 0
        else:
            self._iactive += 1

        if self._iactive not in self._upds.keys():
            print(val, self._thresh)
            out.write_log("Maximum energy difference exceeded at highest level of nuclear integrator.\nTerminating.")
            exit()