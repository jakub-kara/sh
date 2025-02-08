import numpy as np
from .meta import Factory
from .constants import convert

class Timestep(metaclass = Factory):
    def __init__(self, *, dt, steps, **config):
        self._end = convert(config["tmax"], "au")
        self.time = 0
        self.step = 0
        self.dts = np.zeros(steps)
        self.dts[:] = convert(dt, "au")

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
        return self.time > self._end

    def next_step(self):
        self.time += self.dt
        self.step += 1


    def save_state(self):
        pass

class Constant(Timestep, key = "const"):
    pass

class Half(Timestep, key = "half"):
    def __init__(self, **config):
        super().__init__(**config)
        self.maxdt = self.dt
        self.maxit = config.get("max_depth", 10)
        self._enthresh = convert(config.get("enthresh", 1000), "au")
        self.it = 0

    def validate(self, val: float):
        return val < self._enthresh

    def success(self):
        if self.dt < self.maxdt:
            self.it -= 1
            self.dt *= 2

    def fail(self):
        if self.it >= self.maxit:
            raise RuntimeError("Maximum timestep halving depth exceeded. Terminating.")
        self.dt /= 2
        self.it += 1

