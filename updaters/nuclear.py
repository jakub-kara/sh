from typing import Callable
from .updaters import Updater
from classes.meta import Factory
from classes.molecule import Molecule

class NuclearUpdater(Updater, metaclass = Factory):
    def update(self, mols: list[Molecule], dt: float, fun: Callable, **kwargs):
        pass
