import numpy as np
from scipy.linalg import expm
from .meta import Selector
from .constants import convert
from .molecule import Molecule
from .out import Output as out


class Timestep(Selector):
    def __init__(self, *, dt, steps, tmax, **kwargs):
        self._end = convert(tmax, "au")
        self.time = 0
        self.step = 0
        self.dts = np.zeros(steps)
        self.dts[:] = convert(dt, "au")

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
        self.dt = dt

    def validate(self, mols: list[Molecule]):
        self.success = True

    def step_success(self):
        self.save_nupd()

    def step_fail(self):
        pass

    def next_step(self):
        self.time += self.dt
        self.step += 1

    # def adjust(self, *, tmax, **kwargs):
    #     self._end = convert(tmax, "au")
    #     CompositeIntegrator().set_state(**self._nupd)

    # def save_nupd(self):
    #     self._nupd = CompositeIntegrator().get_state()

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
        self._maxdt = self.dt
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

class TDC(Timestep):
    key = "tdc"

    def __init__(self, **config):
        super().__init__(**config)
        self._maxdt = self.dt
        self._eps = config.get("eps", 1e-5)
        self._eta = config.get("eta", 1e-3)
        self._tmpdt = self.dt

        self._past = np.full(3, np.nan)

    def _shift(self, arr, val):
        arr[:-1] = arr[1:]
        arr[-1] = val

    def validate(self, mols: list[Molecule]):
        self.success = True

        mol = mols[-1]
        diff = 0
        for i in range(mol.n_states):
            for j in range(i):
                diff += (np.abs(mol.coeff_s[i])**2 + np.abs(mol.coeff_s[j])**2) * np.abs(mol.nacdt_ss[i,j])
        self._shift(self._past, diff)

        der2 = 0
        self._tmpdt = self.dt
        if not np.isnan(self._past).any():
            h1 = self.dts[-1]
            h2 = self.dts[-2]
            der2 = self._past[-3] / (h1 * (h1 + h2))
            der2 -= self._past[-2] / (h1 * h2)
            der2 += self._past[-1] / (h2 * (h1 + h2))
            der2 *= 2 / (h1 + h2)

            self._tmpdt = self._eta / np.cbrt(np.abs(der2)) * self._maxdt
            print(self._eta / np.cbrt(np.abs(der2)) * self._maxdt)

        # print(np.arccos(np.abs(np.vdot(mol.coeff_s, mols[-2].coeff_s))))
        # self._tmpdt = self._eta / (diff + self._eps)
        # self._tmpdt = self._eta / np.log(diff / self._eps + 1)

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self.dt} {np.real(diff)} {der2}")

    def step_success(self):
        ratio = self._tmpdt / self.dt
        ratio = min(ratio, 2)
        ratio = max(ratio, 0.5)
        self._push_dt(min(self._maxdt, self.dt * ratio))

        with open("ts.dat", "a") as f:
            f.write(f" {self.dt}\n")

class PID(Timestep):
    key = "pid"

    def __init__(self, **config):
        super().__init__(**config)
        self._maxdt = self.dt
        self._alpha = config.get("alpha", 0.2)
        self._beta = config.get("beta", 0.1)
        self._delta = config.get("delta", 5e-3)
        self._eps = config.get("eps", 1e-8)
        self._prev = self._eps

        self._fact = 1

    def validate(self, mols: list[Molecule]):
        self.success = True

        def flux(mol: Molecule):
            tot = 0
            mol = mols[-1]
            for i in range(mol.n_states):
                for j in range(i):
                    tot += np.abs(np.real(mol.coeff_s[i].conj() * mol.coeff_s[j] * mol.nacdt_ss[i,j]))
            return tot

        def dist(mol1: Molecule, mol2: Molecule):
            return np.arccos(np.abs(np.vdot(mol1.coeff_s, mol2.coeff_s)))

        # diff = dist(mols[-1], mols[-2])
        mol = mols[-1]
        diff = self.dt * np.abs(np.einsum("i,j,ij->", mol.coeff_s.conj(), mol.coeff_s, -1j*mol.nacdt_ss).real)
        print(diff)
        curr = diff / self._delta + self._eps

        # self._fact = curr**self._alpha
        self._fact = 1 / curr**self._alpha * (self._prev / curr)**self._beta
        self._fact = min(max(self._fact, 0.5), 2)
        self._prev = curr

        with open("ts.dat", "a") as f:
            f.write(f"{self.time + self.dt} {np.real(diff)} {self._fact}")

    def step_success(self):
        self.dt = min(self._fact * self.dt, self._maxdt)
        if self.time + self.dt > self._end:
            self.dt = self._end - self.time

        with open("ts.dat", "a") as f:
            f.write(f" {self.dt}\n")