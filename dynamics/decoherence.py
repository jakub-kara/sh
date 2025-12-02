import numpy as np

from classes.config import Config, get_cfg
from classes.meta import Singleton
from classes.molecule import Molecule
from classes.timestep import Timestep
from updaters.base import Updater, UpdateResult

@Config("decoherence", default="none", path=Config.DYNAMICS, valid=Config.STR)
class Decoherence(Updater, metaclass = Singleton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out = None

    def new_result(self, mol):
        self.out = UpdateResult(mol.coeff_s, self.substeps)

    def no_update(self, mols: list[Molecule], ts: Timestep):
        self.out.fill()

@Config.Option("none", "decoherence")
class BlankDecoherence(Decoherence):
    def update(self, mols, ts):
        self.no_update(mols, ts)

@Config.Option("edc", "decoherence")
@Config("edc_const", default=0.1, path=Config.DYNAMICS, valid=Config.FLOAT)
class EDCDecoherence(Decoherence):
    def __init__(self, **config):
        super().__init__(**config)
        self.c = get_cfg(self, "edc_const", config)

    def update(self, mols: list[Molecule], ts: Timestep):
        mol = mols[-1]
        out = self.out
        kin_en = mol.kinetic_energy()
        for s in range(mol.n_states):
            if s == mol.active:
                continue
            else:
                decay_rate = 1/np.abs(mol.ham_eig_ss[s,s] - mol.ham_eig_ss[mol.active, mol.active]) * (1 + self.c/kin_en)
                out.out = out.inp * np.exp(-ts.dt/decay_rate)

        inac_pop = np.sum(np.abs(out.out)**2) - np.abs(out.out[mol.active])**2
        out.out[mol.active] *= np.sqrt(1 - inac_pop) / np.abs(out.out[mol.active])

# TODO: GONT, INT, ODC