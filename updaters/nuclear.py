from .base import Updater, UpdateResult
from classes.meta import Selector
from classes.molecule import Molecule

class NuclearUpdater(Updater, Selector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out = None
        self._ready = True

    def new_result(self, mol: Molecule, *args, **kwargs):
        self.out = UpdateResult(mol, self.substeps)

    def no_update(self, *args, **kwargs):
        raise RuntimeError(f"Updates are mandatory by {self.__class__.__name__}.")
