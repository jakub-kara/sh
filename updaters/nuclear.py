from typing import Callable
from .updaters import Updater, UpdateResult
from classes.meta import SingletonFactory
from classes.molecule import Molecule

class NuclearUpdater(Updater, metaclass = SingletonFactory):
    def update(self, mols: list[Molecule], dt: float, fun: Callable, **kwargs):
        pass
