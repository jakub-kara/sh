import numpy as np
from .base import Updater, UpdateResult
from classes.config import Config, get_cfg
from classes.constants import convert
from classes.format import *
from classes.meta import Singleton
from classes.molecule import Molecule
from classes.out import FileManager as fm
from classes.timestep import Timestep

@Config("nuc", valid=Config.STR)
class NuclearUpdater(Updater):
    def is_ready(self, step):
        return True

    def new_result(self, mol: Molecule, *args, **kwargs):
        self.result = UpdateResult(mol, self.substeps)

    def no_update(self, *args, **kwargs):
        raise RuntimeError(f"Updates are mandatory by {self.__class__.__qualname__}.")


@Config("nuc_upd", default="vv", path=Config.NUCLEAR, valid=Config.LIST[Config.STR])
@Config("eshift", default=1e10, path=Config.NUCLEAR, valid=Config.POSITIVE(Config.UNIT))
@Config("edrift", default=1e10, path=Config.NUCLEAR, valid=Config.POSITIVE(Config.UNIT))
class CompositeIntegrator(Updater, metaclass = Singleton):
    def __init__(self, **config):
        super().__init__(**config)
        self._iactive = 0
        self._count = 0
        self._eshift = convert(get_cfg(self, "eshift", config), "au")
        self._edrift = convert(get_cfg(self, "edrift", config), "au")
        self._upds: dict[int, NuclearUpdater] = {}
        self._refen = None

        nuc_upd = get_cfg(self, "nuc_upd", config)
        if not isinstance(nuc_upd, (list, tuple)):
            nuc_upd = [nuc_upd]

        for i, upd in enumerate(nuc_upd):
            self._upds[i] = get_cfg(NuclearUpdater, "nuc", {"nuc": upd})()

        base = self._upds[0]
        if base.steps > 1:
            self._upds[-1] = get_cfg(NuclearUpdater, "nuc", {"nuc": upd})()

        self.to_init()

    @property
    def steps(self):
        return max(upd.steps for upd in self._upds.values())

    @property
    def active(self):
        return self._upds[self._iactive]

    def new_result(self, mol):
        self.result = UpdateResult(mol, self.active.substeps)

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

    def run(self, mols: list[Molecule], ts: Timestep):
        if self._refen is None:
            self._refen = mols[-1].total_energy()

        self.new_result(mols[-1])
        self._set_active()
        self._count += 1
        self.active.run(mols, ts)
        self.result = self.active.result

        self.validate(mols + [self.result.out], ts)

    def validate(self, mols: list[Molecule], ts: Timestep):
        edrift = np.abs(mols[-1].total_energy() - self._refen)
        if edrift > self._edrift:
            fm.log("Maximum energy drift exceeded.")
            fm.log(f"Tolerance: {cfmtu(self._edrift, 'au', 'ev', Size.M)}")
            fm.log(f"Actual:    {cfmtu(edrift, 'au', 'ev', Size.M)}")
            fm.log("Terminating.")
            exit()

        eshift = np.abs(mols[-1].total_energy() - mols[-2].total_energy())
        self.success = eshift <= self._eshift
        if self.success:
            self._iactive = 0
        else:
            self._iactive += 1

        if self._iactive not in self._upds:
            fm.log("Maximum energy shift exceeded at highest level of nuclear integrator.")
            fm.log(f"Tolerance: {cfmtu(self._eshift, 'au', 'ev', Size.M)}")
            fm.log(f"Actual:    {cfmtu(eshift, 'au', 'ev', Size.M)}")
            fm.log("Terminating.")
            exit()