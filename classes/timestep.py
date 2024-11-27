import numpy as np
from .meta import Factory

class Timestep(metaclass = Factory):
    def __init__(self, *, dt, steps, **config):
        self.dts = np.zeros(steps)
        self.dts[:] = dt

    @property
    def dt(self):
        return self.dts[-1]

    @dt.setter
    def dt(self, val: float):
        self.dts[-1] = val

    def validate(self, cond: bool):
        return True

    def success(self):
        pass

    def fail(self):
        pass


class Constant(Timestep, key = "const"):
    pass

class Half(Timestep, key = "half"):
    def __init__(self, **config):
        super().__init__(**config)
        self.maxdt = self.dt
        self.maxit = config.get("max_depth", 10)
        self.it = 0

    def validate(self, cond: bool):
        return cond

    def success(self):
        if self.dt < self.maxdt:
            self.it -= 1
            self.dt *= 2

    def fail(self):
        if self.it >= self.maxit:
            raise RuntimeError("Maximum timestep halving depth exceeded. Terminating.")
        self.dt /= 2
        self.it += 1

