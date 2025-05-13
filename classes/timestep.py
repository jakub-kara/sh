import numpy as np
from scipy.linalg import expm
from scipy.interpolate import lagrange
from .meta import Selector
from .constants import convert
from .molecule import Molecule
from .out import Output as out


class Timestep(Selector):
    def __init__(self, *, dt, steps, tmax, **config):
        self._end = convert(tmax, "au")
        self.time = 0
        self.step = 0
        self.dts = np.zeros(steps)
        self.dts[:] = convert(dt, "au")
        self._maxdt = config.get("maxdt", self.dt)
        self._mindt = config.get("mindt", 0)
        self._fact = 1
        self._lim = config.get("lim", 2)
        if self._lim < 1:
            self._lim = 1 / self._lim
        self.success = False

        self._nupd: dict = None

    @property
    def finished(self):
        return self.time >= self._end

    @property
    def dt(self):
        return self.dts[-1]

    @dt.setter
    def dt(self, val: float):
        self.dts[-1] = val

    def _push_dt(self, dt: float):
        self.dts[:-1] = self.dts[1:]
        self.dt = max(min(dt, self._maxdt), self._mindt)

        # TODO: negative timesteps?

    def _minmax(self):
        self._fact = min(max(self._fact, 1/self._lim), self._lim)

    def validate(self, mols: list[Molecule]):
        self.success = True

    def step_success(self):
        self.save_nupd()

    def step_fail(self):
        pass

    def next_step(self):
        self.time += self.dt
        self.step += 1

    def adjust(self, *args, **kwargs):
        pass

    def save_nupd(self, *args, **kwargs):
        pass

class Constant(Timestep):
    key = "const"

class Half(Timestep):
    key = "half"

    def __init__(self, **config):
        super().__init__(**config)
        self._maxit = config.get("max_depth", 10)
        self._enthresh = convert(config.get("enthresh", 1000), "au")
        self._it = 0
        self._cumdt = 0

    def validate(self, mols: list[Molecule]):
        self.success = np.abs(mols[-1].total_energy() - mols[-2].total_energy()) < self._enthresh

    def _iter_to_dt(self):
        self.dt = self._maxdt / 2**self._it

    def step_success(self):
        super().step_success()
        if self.dt < self._maxdt:
            self._it -= 1
        self._iter_to_dt()

        self._cumdt += self.dt
        if self._cumdt >= self._maxdt:
            self.dt -= self._cumdt - self._maxdt
            self._cumdt = 0

        print("ITER: ", self._it)
        print("DT:   ", self.dt)

    def step_fail(self):
        if self._it >= self._maxit:
            msg = "Maximum timestep halving depth exceeded. Terminating."
            out.write_log(msg)
            raise RuntimeError(msg)
        self._it += 1

        self._iter_to_dt()
        print("ITER: ", self._it)
        print("DT:   ", self.dt)

class Proportional(Timestep):
    key = "prop"

    def __init__(self, **config):
        super().__init__(**config)
        self._alpha = config.get("alpha", 1)
        self._delta = config.get("eps", 1e-8)
        self._eta = config.get("eta", 1e-2)

    def validate(self, mols: list[Molecule]):
        self.success = True

        mol = mols[-1]
        diff = 0
        for i in range(mol.n_states):
            for j in range(i):
                diff += (np.abs(mol.coeff_s[i])**2 + np.abs(mol.coeff_s[j])**2) * np.abs(mol.nacdt_ss[i,j])

        self._fact = self._eta / (diff**self._alpha + self._delta) / self.dt

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self.dt} {np.real(diff)} {self._fact}")

    def step_success(self):
        self._minmax()
        self._push_dt(self.dt * self._fact)

        with open("ts.dat", "a") as f:
            f.write(f" {self.dt}\n")

class Hairer(Timestep):
    # doi: 10.1137/040606995
    key = "hairer"

    def __init__(self, **config):
        super().__init__(**config)
        self._alpha = config.get("alpha", 0.5)
        self._delta = config.get("delta", 1e-8)
        self._eps = config.get("eps", self.dt)
        self._rho = 1

    def _ctrl(self, mols: list[Molecule]):
        tdc = lambda mol: np.abs(mol.nacdt_ss[0,1])
        mol = mols[-1]
        res = self._alpha * (tdc(mol) - tdc(mols[-2])) / self.dt * tdc(mol)
        res /= tdc(mol)**2 + self._delta
        return res

    def validate(self, mols):
        self.success = True

        self._fact = 1 + self._eps * self._ctrl(mols) / self._rho
        self._minmax()
        self._rho *= self._fact

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self._eps / self._rho} {mols[-1].nacdt_ss[0,1]} {self._ctrl(mols)}")

    def step_success(self):
        self._push_dt(self._eps / self._rho)

        with open("ts.dat", "a") as f:
            f.write(f" {self.dt}\n")
