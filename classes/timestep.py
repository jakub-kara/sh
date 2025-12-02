import numpy as np
from scipy.interpolate import CubicSpline
from .config import Config, get_cfg
from .constants import convert
from .format import *
from .molecule import Molecule
from .out import FileManager as fm

@Config("killafter", default=1e10, path=Config.DYNAMICS, valid=Config.LIST[Config.UNIT])
class StateTracker:
    def __init__(self, **config):
        temp = get_cfg(self, "killafter", config)
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
            fm.log(f"Time on state {self._state} exceeded limits.")
            fm.log(f"Time spent on state {self._state}: {cfmtu(self._elapsed, 'au', 'fs')}")
            fm.log(f"Limit on state {self._state}:      {cfmtu(self._killafter.get(self._state, 1e10), 'au', 'fs')}")
            fm.log("Terminating.")
            exit()

@Config("timestep", default="const", path=Config.DYNAMICS, valid=Config.STR)
@Config("dt", path=Config.DYNAMICS, valid=Config.LIST[Config.UNIT])
@Config("keep", default=1, valid=Config.POSITIVE(Config.INT))
@Config("itime", default=0, path=Config.DYNAMICS, valid=Config.UNIT)
@Config("ftime", default=None, path=Config.DYNAMICS, valid=Config.NONE | Config.UNIT)
@Config("istep", default=0, path=Config.DYNAMICS, valid=Config.INT)
@Config("fstep", default=None, path=Config.DYNAMICS, valid=Config.NONE | Config.INT)
class Timestep:
    def __init__(self, **config):
        self._itime = convert(get_cfg(self, "itime", config), "au")
        self._ftime = convert(get_cfg(self, "ftime", config), "au")
        self.time = self._itime

        self._istep = get_cfg(self, "istep", config)
        self._fstep = get_cfg(self, "fstep", config)
        if self._fstep is not None and self._istep > self._fstep:
            self._istep, self._fstep = self._fstep, self._istep
            fm.log("Warning: *istep* and *fstep* swapped to ensure that *istep* < *fstep*.")
        self.step = self._istep

        dt = get_cfg(self, "dt", config)
        if not isinstance(dt, (list, tuple)):
            dt = [dt, dt]
        self._mindt, self._maxdt = sorted([convert(x, "au") for x in dt])
        n_steps = max(get_cfg(self, "keep", config), 1)
        self.dts = np.full(n_steps, self._maxdt)

        self._fact = 1
        self.success = False
        self._nupd: dict = None

    @property
    def finished(self):
        step = self._fstep is None or self.step < self._fstep
        time = self._ftime is None or self.time * self.dt < self._ftime * self.dt
        return not (step and time)

    @property
    def dt(self):
        return self.dts[-1]

    @dt.setter
    def dt(self, val: float):
        self.dts[-1] = val

    @property
    def mindt(self):
        return self._mindt

    @property
    def maxdt(self):
        return self._maxdt

    def _push_dt(self, dt: float):
        self.dts[:-1] = self.dts[1:]
        self.dt = dt

    def _smooth_dt(self, dt: float):
        a = 2
        return -1/a * np.log(np.exp(-a*self._maxdt) + 1 / (np.exp(a * self._mindt) + np.exp(a * dt)))

    def _clip_local(self, dt: float):
        lim = 10
        fact = dt / self.dt
        fact = min(max(fact, 1/lim), lim)
        return self.dt * fact

    def _clip_global(self, dt: float):
        return max(min(dt, self._maxdt), self._mindt)

    def _process_dt(self, dt: float):
        dt = self._clip_local(dt)
        dt = self._clip_global(dt)
        self._push_dt(dt)

    def validate(self, mols: list[Molecule]):
        self.success = True

    def step_success(self, mols: list[Molecule]):
        self.save_nupd()

    def step_fail(self, mols: list[Molecule]):
        pass

    def next_step(self):
        self.time += self.dt
        self.step += 1

    def save_nupd(self, *args, **kwargs):
        pass

@Config.Option("const", "timestep")
class Constant(Timestep):
    pass

@Config.Option("half", "timestep")
@Config("maxdepth", default=10, path=Config.DYNAMICS, valid=Config.INT)
@Config("enthresh", default="0.1 ev", path=Config.DYNAMICS, valid=Config.POSITIVE(Config.UNIT))
@Config("ovlthresh", default=0.95, path=Config.DYNAMICS, valid=Config.POSITIVE(Config.FLOAT))
class Half(Timestep):
    key = "half"

    def __init__(self, **config):
        super().__init__(**config)
        self._maxit = get_cfg(self, "maxdepth", config)
        self._enthresh = convert(get_cfg(self, "enthresh", config), "au")
        self._ovlthresh = get_cfg(self, "ovlthresh", config)
        self._it = 0

        self._diff = 0
        self._ovl = None

    def validate(self, mols: list[Molecule]):
        self._diff = np.abs(mols[-1].total_energy() - mols[-2].total_energy())
        encond = self._diff <= self._enthresh

        ovlcond = True
        ovl = mols[-1].ovlp_ss
        if not np.any(np.isnan(ovl)):
            temp = np.diag(ovl)
            ovlcond = np.all(temp >= self._ovlthresh)
            self._ovl = np.min(temp)

        self.success = encond and ovlcond

    def _iter_to_dt(self):
        self.dt = self._maxdt / 2**self._it

    def step_success(self, mols: list[Molecule]):
        super().step_success()
        if self._it > 0:
            self._it -= 1
        self._iter_to_dt()

    def step_fail(self, mols: list[Molecule]):
        if self._it >= self._maxit:
            fm.log("Maximum timestep halving depth exceeded.")
            fm.log("Terminating.")
            exit()
        self._it += 1

        fm.log("Step rejected.")
        fm.log(f"Energy difference: {cfmtu(self._diff, 'au', 'ev', Size.S)}.")
        if self._ovl is not None:
            fm.log(f"Overlap: {fmt(self._ovl, Size.S)}.")

        self._iter_to_dt()

@Config("monitor", default="pop", path=Config.DYNAMICS)
class MonitorFunc:
    def fun(self, mols: list[Molecule], ts: Timestep):
        raise NotImplementedError

    def der(self, mols: list[Molecule], ts: Timestep):
        raise NotImplementedError

    def __call__(self, mols: list[Molecule], ts: Timestep):
        return self.fun(mols, ts)

    def check(self, mols: list[Molecule], ts: Timestep):
        return None

@Config.Option("tdc", "monitor")
class TDCFunc(MonitorFunc):
    def fun(self, mols):
        mol = mols[-1]
        tot = 0
        for i in range(mol.n_states):
            for j in range(i):
                tot += (np.abs(mol.coeff_s[i])**2 + np.abs(mol.coeff_s[j])**2) * np.abs(mol.nacdt_ss[i,j])

        # return 0.5*np.sum(np.abs(mol.nacdt_ss))
        return tot

@Config.Option("pos", "monitor")
class PosFunc(MonitorFunc):
    def fun(self, mols):
        mol = mols[-1]
        return np.sqrt(0.5 * np.sum(mol.mom_ad**2 / mol.mass_a[:,None]))

@Config.Option("phs", "monitor")
class PhsFunc(MonitorFunc):
    def fun(self, mols):
        mol = mols[-1]
        return np.sqrt(0.5 * np.sum(mol.mom_ad**2 / mol.mass_a[:,None]) + np.sum(mol.force_ad**2 / mol.mass_a[:,None]))

@Config.Option("pop", "monitor")
class PopFunc(MonitorFunc):
    def fun(self, mols: list[Molecule], ts: Timestep):
        def absmax(x, y):
            if np.abs(x) > np.abs(y):
                return x
            else:
                return y

        mol = mols[-1]
        act = mol.active
        tot = 0
        for i in range(mol.n_states):
            if i == act:
                continue

            a = 2 * absmax(mol.nacdt_ss[act,i], mol.nacdt_ss[act,i] + (mol.nacdt_ss[act,i] - mols[-2].nacdt_ss[act,i]) / ts.dt * ts.maxdt)
            b = mol.ham_eig_ss[i,i] - mol.ham_eig_ss[act,act]
            c = np.sqrt(a**2 + b**2)
            h = ts.maxdt
            sz = mol.bloch_n3[i,2]
            sx = np.sqrt(1 - sz**2)
            sy = np.sqrt(1 - sz**2)
            tot += 0.5 * np.abs(-a/c * np.sin(c*h) * sx - a*b/c**2 * (1 - np.cos(c*h)) * sy + (np.cos(c*h) + b**2/c**2 * (1 - np.cos(c*h))) * sz - sz)
            # tot = max(tot, 0.5 * np.abs(-a/c * np.sin(c*h) * sx - a*b/c**2 * (1 - np.cos(c*h)) * sy + (np.cos(c*h) + b**2/c**2 * (1 - np.cos(c*h))) * sz - sz))
        return tot

    def check(self, mols, ts):
        # return 0.25 * np.nanmax(np.abs(mols[-1].bloch_n3[:,2] - mols[-2].bloch_n3[:,2]))
        return 0.25 * np.nansum(np.abs(mols[-1].bloch_n3[:,2] - mols[-2].bloch_n3[:,2]))

@Config.Option("mas", "monitor")
class Massey(MonitorFunc):
    def fun(self, mols: list[Molecule], ts: Timestep):
        def absmax(x, y):
            if np.abs(x) > np.abs(y):
                return x
            else:
                return y

        mol = mols[-1]
        act = mol.active
        tot = 0
        for i in range(mol.n_states):
            if i == act:
                continue

            a = absmax(mol.nacdt_ss[act,i], mol.nacdt_ss[act,i] + (mol.nacdt_ss[act,i] - mols[-2].nacdt_ss[act,i]) / ts.dt * ts.maxdt)
            b = np.abs(mol.ham_eig_ss[i,i] - mol.ham_eig_ss[act,act])
            tot = max(tot, np.abs(a) / b)
        return tot

    def check(self, mols, ts):
        mol = mols[-1]
        act = mol.active
        tot = 0
        for i in range(mol.n_states):
            tot = max(tot, np.abs(mol.nacdt_ss[i, act]) / np.abs(mol.ham_eig_ss[i,i] - mol.ham_eig_ss[act, act]))
        return tot

@Config.Option("symp", "timestep")
@Config("alpha", default=1, path=Config.DYNAMICS)
@Config("delta", default=1e-10, path=Config.DYNAMICS)
class Symplectic(Timestep):
    # doi: 10.1137/040606995
    def __init__(self, **config):
        super().__init__(**config)
        self._alpha = get_cfg(self, "alpha", config)
        self._delta = get_cfg(self, "delta", config)
        self._eps = self.dt
        self._rho = 1

        self._func = get_cfg(self, "monitor", config)()
        self._vals = np.full((3,2), np.nan)


    def _ctrl(self, mols: list[Molecule]):
        self._vals[:-1] = self._vals[1:]
        self._vals[-1,0] = self.dt

        self._vals[-1,1] = self._func(mols[-1])
        if np.any(np.isnan(self._vals)) or np.any(self._vals[:,1] < self._delta):
            return 0

        times = np.cumsum(self._vals[:,0])
        times -= times[-1]
        spl = CubicSpline(times, np.log(self._vals[:,1]))

        temp = self._func(mols[-1])
        res = self._alpha * spl(0, 1)

        res = self._alpha * np.sum(temp * spl(0, 1)) / (np.sum(temp**2) + self._delta)
        res = self._alpha / self.dt * np.sum(temp * (temp - self._func(mols[-2]))) / (np.sum(temp**2) + self._delta)

        print("FUNC: ", self._vals[-1,1])
        print("CTRL: ", res)
        return res

    def step_success(self, mols):
        self._rho += self._eps * self._ctrl(mols)
        dt = self._eps / self._rho
        self._process_dt(dt)
        self._rho = self._eps / self.dt

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self.dt} {self.dt}\n")

@Config.Option("pid", "timestep")
@Config("kp", default=0, path=Config.DYNAMICS)
@Config("ki", default=0, path=Config.DYNAMICS)
@Config("kd", default=0, path=Config.DYNAMICS)
@Config("intlim", default=1, path=Config.DYNAMICS)
class PID(Timestep):
    def __init__(self, **config):
        super().__init__(**config)
        self._kp = get_cfg(self, "kp", config)
        self._ki = get_cfg(self, "ki", config)
        self._kd = get_cfg(self, "kd", config)

        self._intlim = get_cfg(self, "intlim", config)
        self._intsum = 0
        self._prev = None
        self._target = config["target"]

        self._mon = get_cfg(self, "monitor", config)()

    def _get_multiplier(self, mols: list[Molecule]):
        err = (self._target - self._mon(mols, self)) / self._target
        print(self._mon(mols, self))
        print(err)
        tp = self._kp * err

        # Integral term (with clamping to prevent windup)
        self._intsum += err
        self._intsum = max(-self._intlim, min(self._intsum, self._intlim)) # Clamping
        ti = self._ki * self._intsum

        # Derivative term
        td = self._kd * (err - self._prev) if self._prev is not None else 0

        self._prev = err

        # The PID output is the total adjustment
        out = tp + ti + td

        # We apply the output as a multiplier to the current step size
        return out + 1

    def step_success(self, mols):
        fact = self._get_multiplier(mols)
        print(fact)
        dt = self.dt * fact
        print(dt)
        self._process_dt(dt)

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self.dt} {self.dt}\n")

@Config.Option("prop", "timestep")
@Config("threshval", path=Config.DYNAMICS, valid=Config.LIST[Config.UNIT])
@Config("delta", default=1e-10, path=Config.DYNAMICS, valid=Config.POSITIVE(Config.UNIT))
@Config("gamma", default=0.8, path=Config.DYNAMICS, valid=Config.POSITIVE(Config.FLOAT))
@Config.Init(MonitorFunc)
class Proportional(Timestep):
    def __init__(self, **config):
        super().__init__(**config)
        self._delta = get_cfg(self, "delta", config)
        self._qmax, self._qmin = sorted([convert(val, "au") for val in get_cfg(self, "threshval", config)])
        self._eta = self._qmin
        self._alpha = - np.log(self._maxdt / self._mindt) / np.log(self._qmax / self._qmin)

        self._gamma = get_cfg(self, "gamma", config)
        self._fact = 1
        self._pred = 1e10

        self._mon: MonitorFunc = get_cfg(MonitorFunc, "monitor", config)()

        with open("0/ts.dat", "w") as f:
            pass

    def validate(self, mols):
        tot = self._mon.check(mols, self)
        self._fact = self._pred / (tot + self._delta)
        self.success = self._fact >= 1 or tot <= self._qmax or self.dt <= self._mindt
        print(self._pred, tot)
        # self.success = self._fact >= 1 or self.dt <= self._mindt
        # self.success = True

    def step_fail(self, mols):
        self._process_dt(self.dt * self._gamma * self._fact)
        print("REJECTED")

    def step_success(self, mols):
        # self._pred = 1 / self._gamma * self._mon(mols, self)
        self._pred = self._mon(mols, self)
        dt = self._mindt * (self._eta / (self._pred + self._delta))**self._alpha
        self._process_dt(dt)

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self.dt} {self.dt}\n")