import numpy as np
from .meta import Selector
from .constants import convert
from updaters.composite import CompositeIntegrator

class Timestep(Selector):
    def __init__(self, *, dt, steps, tmax, **kwargs):
        self._end = convert(tmax, "au")
        self.time = 0
        self.step = 0
        self.dts = np.zeros(steps)
        self.dts[:] = convert(dt, "au")

        self._nupd: dict = None

    @property
    def dt(self):
        return self.dts[-1]

    @dt.setter
    def dt(self, val: float):
        self.dts[-1] = val

    def validate(self, val):
        return True

    def success(self):
        pass

    def fail(self):
        pass

    @property
    def finished(self):
        return self.time >= self._end

    def next_step(self):
        self.time += self.dt
        self.step += 1

    def adjust(self, *, tmax, **kwargs):
        self._end = convert(tmax, "au")
        CompositeIntegrator().set_state(**self._nupd)

    def save_nupd(self):
        self._nupd = CompositeIntegrator().get_state()

class Constant(Timestep):
    key = "const"

class Half(Timestep):
    key = "half"

    def __init__(self, **config):
        super().__init__(**config)
        self.maxdt = self.dt
        self.maxit = config.get("max_depth", 10)
        self._enthresh = convert(config.get("enthresh", 1000), "au")
        self._it = 0

    def validate(self, val: float):
        return val < self._enthresh

    def success(self):
        if self.dt < self.maxdt:
            self._it -= 1
            self.dt *= 2

    def fail(self):
        if self._it >= self.maxit:
            raise RuntimeError("Maximum timestep halving depth exceeded. Terminating.")
        self.dt /= 2
        self._it += 1
