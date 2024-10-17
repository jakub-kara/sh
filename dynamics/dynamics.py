import numpy as np
from abc import ABC, abstractmethod
from classes.molecule import Molecule
from electronic.electronic import ESTProgram
from classes.constants import Constants


# TODO: move Control to Dynamics
class Dynamics(ABC):
    def __init__(self, **config: dict):
        self._name = config["name"]
        self._type = config["method"].upper()

        tconv = {
            "fs": 1/Constants.au2fs,
            "au": 1,
        }[config.get("tunit", "au")]

        self._dt = config["dt"] * tconv
        self._dtmax = self._dt
        self._end = config["tmax"] * tconv
        self._time = 0
        self._step = 0
        self._enthresh = config.get("enthresh", 1000)

        self._stepok = True

        super().__init__()


    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def is_finished(self):
        return self._time > self._end

    @property
    def name(self):
        return self._name

    @property
    def dt(self):
        return self._dt

    @property
    def curr_step(self):
        return self._step

    @property
    def curr_time(self):
        return self._time

    @property
    def en_thresh(self):
        return self.en_thresh

    @property
    def step_ok(self):
        return self._stepok

    def next_step(self):
        self._time += self.dt
        self._step += 1

    def check_energy(self, energy_diff):
        self._stepok = energy_diff < self._enthresh

    @abstractmethod
    def calculate_acceleration(self, mol: Molecule):
        pass

    @abstractmethod
    def potential_energy(self, mol: Molecule):
        pass

    @abstractmethod
    def prepare_traj(self, mol: Molecule):
        pass

    # might have a better name
    @abstractmethod
    def adjust_nuclear(self, mol: Molecule):
        pass

    @abstractmethod
    def setup_est(self, est: ESTProgram, mode: str = ""):
        pass