from abc import ABC, abstractmethod
from typing import Callable
from classes.molecule import Molecule

class Integrator1(ABC):
    def __init__(self):
        self.name = None
        self.steps = 1
        self.substeps = 1
        self.isub = 1
        self.order = None

    @abstractmethod
    def integrate(self, fun: Callable, dt: float, *args: Molecule) -> Molecule:
        raise NotImplementedError
    
class Integrator2(ABC):
    def __init__(self):
        self.steps = 1
        self.substeps = 1
        self.isub = 1

    @abstractmethod
    def integrate(self, fun: Callable, dt: float, *args: Molecule) -> Molecule:
        raise NotImplementedError