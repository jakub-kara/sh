import numpy as np
from scipy.linalg import expm
from scipy.interpolate import CubicSpline
from .meta import Selector
from .constants import convert
from .molecule import Molecule
from .out import Output as out


class StateTracker:
    def __init__(self, **config):
        temp = config.get("killafter", 1e10)
        if isinstance(temp, dict):
            temp = {int(k): convert(v, "au") for k, v in temp.items()}
        elif isinstance(temp, list):
            temp = {iv: convert(v, "au") for iv, v in enumerate(temp)}
        else:
            temp = {0: convert(temp, "au")}
        self._killafter = temp
        self._elapsed = 0
        self._state = -1

    def count(self, mol: Molecule, dt):
        if not hasattr(mol, "active"):
            return

        if mol.active == self._state:
            self._elapsed += dt
        else:
            self._elapsed = 0
            self._state = mol.active

        if self._elapsed > self._killafter.get(self._state, 1e10):
            out.write_log(f"Time on state {self._state} exceeded limits.\nTerminating.")
            exit()

class Timestep(Selector):
    def __init__(self, *, dt, steps, tmax, **config):
        self._end = convert(tmax, "au")
        self.time = 0
        self.step = 0
        self.dts = np.zeros(steps)
        self.dts[:] = convert(dt, "au")
        self._maxdt = convert(config.get("maxdt", self.dt), "au")
        self._mindt = convert(config.get("mindt", 0), "au")
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

    def adjust(self, **config):
        end = config.get("tmax", None)
        if end is not None:
            self._end = convert(end, "au")

        dt = config.get("dt", None)
        if end is not None:
            self.dt = convert(dt, "au")

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

        self._diff = 0

    def validate(self, mols: list[Molecule]):
        self._diff = np.abs(mols[-1].total_energy() - mols[-2].total_energy())
        self.success = self._diff < self._enthresh

    def _iter_to_dt(self):
        self.dt = self._maxdt / 2**self._it

    def step_success(self):
        super().step_success()
        if self._it > 0:
            self._it -= 1
        self._iter_to_dt()

        # self._cumdt += self.dt
        # if self._cumdt >= self._maxdt:
        #     self.dt -= self._cumdt - self._maxdt
        #     self._cumdt = 0

        # print("ITER: ", self._it)
        # print("DT:   ", self.dt)

    def step_fail(self):
        if self._it >= self._maxit:
            msg = "Maximum timestep halving depth exceeded. Terminating."
            out.write_log(msg)
            raise RuntimeError(msg)
        self._it += 1

        out.write_log(f"Step rejected with energy difference {convert(self._diff, 'ev'):.6f} eV")


        self._iter_to_dt()
        # print("ITER: ", self._it)
        # print("DT:   ", self.dt)

def _tdc(mol: Molecule):
    return 0.5*np.sum(np.abs(mol.nacdt_ss))
    tot = 0
    for i in range(mol.n_states):
        for j in range(i):
            tot += (np.abs(mol.coeff_s[i])**2 + np.abs(mol.coeff_s[j])**2) * np.abs(mol.nacdt_ss[i,j])
    return tot

def _acc(mol: Molecule):
    return np.sqrt(np.max(np.linalg.norm(mol.acc_ad, axis=1)))

def _vel(mol: Molecule):
    return np.max(np.linalg.norm(mol.vel_ad, axis=1))

def _pos(mol: Molecule):
    return np.abs(mol.pos_ad[0,0])

def _eff(mol: Molecule):
    return np.linalg.norm(mol.eff_nac())

def _grad(mol: Molecule):
    return np.linalg.norm(mol.force_ad)

def _select_func(val):
    return {
        "pos": _pos,
        "vel": _vel,
        "acc": _acc,
        "tdc": _tdc,
        "eff": _eff,
        "grad": _grad,
    }[val]


class Proportional(Timestep):
    key = "prop"

    def __init__(self, **config):
        super().__init__(**config)
        self._alpha = config.get("alpha", 1)
        self._delta = config.get("eps", 1e-8)
        self._eta = config.get("eta", 1e-2)

        self._val = config.get("val", "tdc")
        self._func = _select_func(self._val)

        with open("ts.dat", "w") as f:
            pass

    def validate(self, mols: list[Molecule]):
        self.success = True

        mol = mols[-1]
        self._fact = self._maxdt * (self._eta / (self._func(mol) + self._delta))**self._alpha / self.dt

    def step_success(self):
        self._minmax()
        self._push_dt(self.dt * self._fact)

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self.dt} {self.dt}\n")

class Curvature(Proportional):
    key = "curv"

    def __init__(self, **config):
        super().__init__(**config)
        self._past = [None]*3

    def validate(self, mols):
        self.success = True

        self._past[:-1] = self._past[1:]
        self._past[-1] = self._func(mols[-1])
        if None in self._past:
            self._fact = 1
            return

        times = np.cumsum(self.dts[-3:])
        times -= times[-1]
        spl = CubicSpline(times, self._past)
        der2 = spl(0, 2)
        self._fact = self._maxdt * (self._eta / (np.cbrt(np.abs(der2)) + self._delta))**self._alpha / self.dt

class Hairer(Timestep):
    # doi: 10.1137/040606995
    key = "hairer"

    def __init__(self, **config):
        super().__init__(**config)
        self._alpha = config.get("alpha", 0.5)
        self._delta = config.get("delta", 1e-8)
        self._eps = config.get("eps", self.dt)
        self._rho = 1

        self._val = config.get("val", "tdc")
        self._func = _select_func(self._val)

        with open("ts.dat", "w") as f:
            pass

    def _ctrl(self, mols: list[Molecule]):
        # TDC
        # mol = mols[-1]
        # temp = self._func(mol)
        # res = self._alpha * (temp - self._func(mols[-2])) / self.dt * temp
        # res /= temp**2 + self._delta

        # res = self._alpha / self.dt * np.log(self._func(mols[-1]) / self._func(mols[-2]))

        # VEL
        # res = self._alpha * np.sum(mols[-1].mom_ad * mols[-1].force_ad)

        # GRAD
        # mol = mols[-1]
        # res = self._alpha / self.dt * np.sum(mol.force_ad * (mol.force_ad - mols[-2].force_ad)) / (np.sum(mol.force_ad**2) + self._delta)

        temp = self._func(mols[-1])
        res = self._alpha / self.dt * np.sum(temp * (temp - self._func(mols[-2]))) / (np.sum(temp**2) + self._delta)

        print("FUNC: ", temp)
        print("CTRL: ", res)
        return res

    def validate(self, mols):
        self.success = True

        self._fact = 1 + self._eps * self._ctrl(mols) / self._rho
        self._minmax()
        self._rho *= self._fact

    def step_success(self):
        self._push_dt(self._eps / self._rho)

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self.dt} {self.dt}\n")
