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
    def __init__(self, *, dt, steps = 1, **config):
        self._itime = convert(config.get("itime", 0), "au")
        self._ftime = config.get("ftime", None)
        if self._ftime is not None:
            self._ftime = convert(self._ftime, "au")
        self.time = self._itime

        self._istep = config.get("istep", 0)
        self._fstep = config.get("fstep", None)
        self.step = self._istep


        if not isinstance(dt, (list, tuple)):
            dt = [dt, dt]
        self._mindt, self._maxdt = sorted([convert(x, "au") for x in dt])
        self.dts = np.zeros(steps)
        self.dts[:] = convert(self._maxdt, "au")

        self._fact = 1
        self._lim = config.get("lim", 2)
        if self._lim < 1:
            self._lim = 1 / self._lim

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
        fact = dt / self.dt
        fact = min(max(fact, 1/self._lim), self._lim)
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

class Constant(Timestep):
    key = "const"

class Half(Timestep):
    key = "half"

    def __init__(self, **config):
        super().__init__(**config)
        self._maxit = config.get("max_depth", 10)
        self._enthresh = convert(config.get("enthresh", "0.1 ev"), "au")
        self._ovlthresh = config.get("ovlthresh", 0.95)
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
            msg = "Maximum timestep halving depth exceeded. Terminating."
            out.write_log(msg)
            raise RuntimeError(msg)
        self._it += 1

        msg = f"Step rejected with energy difference {convert(self._diff, 'ev'):.6f} eV"
        if self._ovl is not None:
            msg += f"and overlap {self._ovl}"
        out.write_log(msg)

        self._iter_to_dt()

class MonitorFunc(Selector):
    def fun(self, mols: list[Molecule], ts: Timestep):
        raise NotImplementedError

    def der(self, mols: list[Molecule], ts: Timestep):
        raise NotImplementedError

    def __call__(self, mols: list[Molecule], ts: Timestep):
        return self.fun(mols, ts)

    def check(self, mols: list[Molecule], ts: Timestep):
        return None

class TDCFunc(MonitorFunc):
    key = "tdc"

    def fun(self, mols):
        mol = mols[-1]
        tot = 0
        for i in range(mol.n_states):
            for j in range(i):
                tot += (np.abs(mol.coeff_s[i])**2 + np.abs(mol.coeff_s[j])**2) * np.abs(mol.nacdt_ss[i,j])

        # return 0.5*np.sum(np.abs(mol.nacdt_ss))
        return tot

class PosFunc(MonitorFunc):
    key = "pos"

    def fun(self, mols):
        mol = mols[-1]
        return np.sqrt(0.5 * np.sum(mol.mom_ad**2 / mol.mass_a[:,None]))

class PhsFunc(MonitorFunc):
    key = "phs"

    def fun(self, mols):
        mol = mols[-1]
        return np.sqrt(0.5 * np.sum(mol.mom_ad**2 / mol.mass_a[:,None]) + np.sum(mol.force_ad**2 / mol.mass_a[:,None]))

class PopFunc(MonitorFunc):
    key = "pop"

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
            tot += np.abs(-a/c * np.sin(c*h) * sx - a*b/c**2 * (1 - np.cos(c*h)) * sy + (np.cos(c*h) + b**2/c**2 * (1 - np.cos(c*h))) * sz - sz)
            # tot = max(tot, np.abs(-a/c * np.sin(c*h) * sx - a*b/c**2 * (1 - np.cos(c*h)) * sy + (np.cos(c*h) + b**2/c**2 * (1 - np.cos(c*h))) * sz - sz))
        return 0.5 * tot

    def check(self, mols, ts):
        return 0.5 * np.nansum(np.abs(mols[-1].bloch_n3[:,2] - mols[-2].bloch_n3[:,2]))

class Massey(MonitorFunc):
    key = "mas"

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

# class Curvature(Proportional):
#     key = "curv"

#     def __init__(self, **config):
#         super().__init__(**config)
#         self._vals = np.full((3,2), np.nan)

#     def validate(self, mols):
#         self.success = True

#     def step_success(self, mols):
#         self._vals[:-1] = self._vals[1:]
#         self._vals[-1,0] = self.dt

#         self._vals[-1,1] = self._mon(mols)
#         if np.any(np.isnan(self._vals)):
#             dt = self.dt
#         else:
#             times = np.cumsum(self._vals[:,0])
#             times -= times[-1]
#             spl = CubicSpline(times, self._vals[:,1])
#             der2 = spl(0, 2)
#             dt = self._maxdt * (self._eta / (np.abs(der2)**(1/3) + self._delta))**self._alpha

#         self._process_dt(dt)

#         with open("ts.dat", "a") as f:
#             f.write(f"{self.time + self.dt} {self.dt}\n")

class Symplectic(Timestep):
    # doi: 10.1137/040606995
    key = "symp"

    def __init__(self, **config):
        super().__init__(**config)
        self._alpha = config.get("alpha", 0.5)
        self._delta = config.get("delta", 1e-8)
        self._eps = config.get("eps", self.dt)
        self._rho = 1

        self._func = MonitorFunc.select(config.get("monitor", "tdc"))()
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

class PID(Timestep):
    key = "pid"

    def __init__(self, **config):
        super().__init__(**config)
        self._kp = config.get("kp", 0)
        self._ki = config.get("ki", 0)
        self._kd = config.get("kd", 0)

        self._intlim = config.get("intlim", 1)
        self._intsum = 0
        self._prev = None
        self._target = config["target"]

        self._mon = MonitorFunc.select(config.get("monitor", "tdc"))()

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

class Proportional(Timestep):
    key = "prop"

    def __init__(self, **config):
        super().__init__(**config)
        self._delta = config.get("delta", 1e-8)
        self._qmax, self._qmin = sorted([convert(val, "au") for val in config["vals"]])
        self._eta = self._qmin
        self._alpha = - np.log(self._maxdt / self._mindt) / np.log(self._qmax / self._qmin)

        self._gamma = config.get("gamma", 0.9)
        self._fact = 1
        self._pred = 1e10

        self._mon: MonitorFunc = MonitorFunc.select(config["monitor"])()

        with open("0/ts.dat", "w") as f:
            pass

    def validate(self, mols):
        tot = self._mon.check(mols, self)
        self._fact = self._pred / (tot + self._delta)
        self.success = self._fact >= 1 or tot <= self._qmax or self.dt <= self._mindt
        # self.success = self._fact >= 1 or self.dt <= self._mindt

    def step_fail(self, mols):
        self._process_dt(self.dt * self._gamma * self._fact)
        print("REJECTED")

    def step_success(self, mols):
        self._pred = self._mon(mols, self)
        dt = self._mindt * (self._eta / (self._pred + self._delta))**self._alpha
        self._process_dt(dt)

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self.dt} {self.dt}\n")