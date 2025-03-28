import numpy as np
from scipy.linalg import expm
from functools import cache
from bisect import bisect_right
import h5py
import os
from classes.constants import convert, atomic_widths

def get_dirs(loc=".", cond=lambda x : True):
    return [d for d in os.listdir(loc) if (os.path.isdir(d) and not d.startswith(".") and cond(d))]

class View:
    def __init__(self):
        self.wid = None
        self.mass = None
        # self.wei = None

        self.pos = None
        self.vel = None
        self.acc = None

        self.amp = None
        self.ham = None
        self.grad = None
        self.nac = None
        self.phs = None

    @property
    def nst(self):
        return self.amp.shape[0]

    @property
    def mom(self):
        return self.vel * self.mass[:,None]

    @property
    def tdc(self):
        return np.einsum("ad, ijad -> ij", self.vel, self.nac)

class Cloning:
    def __init__(self, time, parent, child, transfer):
        self.time = time
        self.parent = parent
        self.child = child
        self.trans = transfer

class Traj:
    def __init__(self):
        self.disabled = False

        self.wid = None
        self.mass = None
        self.time = []

        self.pos = []
        self.vel = []
        self.acc = []

        self.amp = []
        self.ham = []
        self.grad = []
        self.nac = []
        self.phs = []

    def is_active(self, time: float):
        return time >= self.time[0] and time <= self.time[-1]

    def read_data(self, file: str, t_start: float = 0):
        with h5py.File(file, "r") as f:
            atoms = f["info/ats"][:].astype("<U2")
            self.wid = np.array([atomic_widths[at] for at in atoms])
            self.mass = f["info/mass"][:]

            keys = [key for key in f.keys()]
            keys.remove("info")
            steps = sorted(map(int, keys))
            for step in steps:
                time = f[f"{step}/time"][()]
                if time < t_start:
                    continue

                self.time.append(time)

                self.pos.append(f[f"{step}/pos"][:])
                self.vel.append(f[f"{step}/vel"][:])
                self.acc.append(f[f"{step}/acc"][:])

                self.amp.append(f[f"{step}/coeff"][:])
                self.ham.append(f[f"{step}/hdiag"][:])
                self.grad.append(f[f"{step}/grad"][:])
                self.nac.append(f[f"{step}/nacdr"][:])
                self.phs.append(f[f"{step}/phase"][()])
        return self

    def to_array(self):
        self.time = np.array(self.time)
        self.wei = np.zeros_like(self.time)

        self.pos = np.array(self.pos)
        self.vel = np.array(self.vel)
        self.acc = np.array(self.acc)

        self.amp = np.array(self.amp)
        self.ham = np.array(self.ham)
        self.grad = np.array(self.grad)
        self.nac = np.array(self.nac)
        self.phs = np.array(self.phs)
        return self

    def _interpolate(self, vals, frac: float):
        return frac * vals[0] + (1 - frac) * vals[1]

    def _slerp(self, vals, frac: float):
        n = vals[0].shape[0]
        temp0 = np.array([np.real(vals[0]), np.imag(vals[0])]).flatten()
        temp1 = np.array([np.real(vals[1]), np.imag(vals[1])]).flatten()
        omega = np.arccos(np.sum(temp0 * temp1))
        temp = np.sin(frac * omega)/np.sin(omega) * temp0 + np.sin((1 - frac) * omega)/np.sin(omega) * temp1
        res = np.zeros_like(vals[0])
        res += temp[:n]
        res += 1j * temp[n:]
        return res

    def create_view(self, time: float):
        if not self.is_active(time):
            raise ValueError(f"Time {time} out of bounds for [{self.time[0]}, {self.time[-1]}]")

        ind = bisect_right(self.time, time)
        frac = (self.time[ind] - time)/(self.time[ind] - self.time[ind-1])

        view = View()
        view.wid = self.wid
        view.mass = self.mass

        view.pos = self._interpolate(self.pos[ind-1:ind+1], frac)
        view.vel = self._interpolate(self.vel[ind-1:ind+1], frac)
        view.acc = self._interpolate(self.acc[ind-1:ind+1], frac)

        # view.amp = self._interpolate(self.amp[ind-1:ind+1], frac)
        # view.amp /= np.sqrt(np.sum(np.abs(view.amp)**2))
        view.amp = self._interpolate(self.amp[ind-1:ind+1], frac)
        view.ham = self._interpolate(self.ham[ind-1:ind+1], frac)
        view.grad = self._interpolate(self.grad[ind-1:ind+1], frac)
        view.nac = self._interpolate(self.nac[ind-1:ind+1], frac)
        view.phs = self._interpolate(self.phs[ind-1:ind+1], frac)

        return view

class Bundle:
    def __init__(self):
        self.disabled = False
        self.trajs: list[Traj] = []
        self.clone: list[Cloning] = []

    def __getitem__(self, key):
        return self.trajs[key]

    @property
    def n_traj(self):
        return len(self.trajs)

    def add_traj(self, traj: Traj):
        self.trajs.append(traj)
        return self

    def read_data(self):
        with open("clone.log") as file:
            for i, line in enumerate(file):
                if i == 0:
                    self.clone.append(Cloning(0, 0, 0, 0))
                    continue
                data = line.strip().split()
                self.clone.append(Cloning(float(data[0]), int(data[2]), int(data[4]), float(data[5])))

        for i, dr in enumerate(get_dirs()):
            os.chdir(dr)
            self.add_traj(Traj().read_data("data/out.h5", self.clone[i].time).to_array())
            os.chdir("..")
        return self

class Ensemble:
    def __init__(self, root: str) -> None:
        os.chdir(root)
        self.bunds: list[Bundle] = []

    def __getitem__(self, key):
        return self.bunds[key]

    @property
    def n_bund(self):
        return len(self.bunds)

    def get_trajs(self, time: float):
        return [traj for bund in self.bunds for traj in bund.trajs if traj.is_active(time)]

    def add_bundle(self, bund: Bundle):
        self.bunds.append(bund)
        return self

    def read_data(self):
        for dr in get_dirs():
            os.chdir(dr)
            self.add_bundle(Bundle().read_data())
            os.chdir("..")
        return self

    def propagate_weights(self, wei_ini, t_ini, t_fin):
        trajs = self.get_trajs(t_ini)
        ntraj = len(trajs)
        wei_fin = np.zeros(ntraj, dtype=np.complex128)
        ovl_ini = np.zeros((ntraj, ntraj), dtype=np.complex128)
        ovl_fin = np.zeros((ntraj, ntraj), dtype=np.complex128)
        ddt_ini = np.zeros((ntraj, ntraj), dtype=np.complex128)
        ddt_fin = np.zeros((ntraj, ntraj), dtype=np.complex128)
        ham_ini = np.zeros((ntraj, ntraj), dtype=np.complex128)
        ham_fin = np.zeros((ntraj, ntraj), dtype=np.complex128)

        view_ini = np.empty(ntraj, dtype=object)
        view_fin = np.empty(ntraj, dtype=object)

        for i, traj in enumerate(trajs):
            view_ini[i] = traj.create_view(t_ini)
            view_fin[i] = traj.create_view(t_fin)

        for i, v1 in enumerate(view_ini):
            for j, v2 in enumerate(view_ini):
                ovl_ini[i,j] = tbf_ovl(v1, v2)
                ddt_ini[i,j] = tbf_ddt(v1, v2)
                ham_ini[i,j] = tbf_ham(v1, v2)

        for i, v1 in enumerate(view_fin):
            for j, v2 in enumerate(view_fin):
                ovl_fin[i,j] = tbf_ovl(v1, v2)
                ddt_fin[i,j] = tbf_ddt(v1, v2)
                ham_fin[i,j] = tbf_ham(v1, v2)

        dt = t_fin - t_ini
        wei_fin[:] = wei_ini
        # print(ddt_ini)
        # print(ham_ini)
        # print(ovl_ini)
        # print(tbf_ddt(view_ini[0], view_ini[1]))
        # print(view_ini[0].amp)
        print(wei_fin)
        print(np.einsum("i,j,ij->", wei_fin.conj(), wei_fin, ovl_ini))
        # print(ovl_ini)
        # print(np.sum(np.abs(wei_fin)**2))

        def ham(frac):
            return (1 - frac) * ham_ini + frac * ham_fin

        def ovl(frac):
            return (1 - frac) * ovl_ini + frac * ovl_fin

        def ddt(frac):
            return (1 - frac) * ddt_ini + frac * ddt_fin

        def func(frac, y):
            return -np.linalg.inv(ovl(frac)) @ (1j*ham(frac) + ddt(frac)) @ y

        k1 = func(0, wei_fin)
        k2 = func(1/2, wei_fin + dt*k1/2)
        k3 = func(1/2, wei_fin + dt*k2/2)
        k4 = func(1, wei_fin + dt*k3)
        # wei_fin += dt*(k1+2*k2+2*k3+k4)/6
        nsteps = 20
        for i in range(nsteps):
            frac = (i + 0.5)/nsteps
            wei_fin = expm(-np.linalg.inv(ovl(frac)) @ (1j*ham(frac) + ddt(frac)) * dt/nsteps) @ wei_fin

        return wei_fin

def main():
    ens = Ensemble(".").read_data()
    # v1 = ens[0][0].create_view(0)
    # v2 = ens[0][1].create_view(0)
    # w0 = np.sqrt(1/(2 * (1 + np.real(tbf_ovl(v1, v2)))))
    # print()
    # wei = np.array([w0, w0])
    wei = np.array([1])
    for i in range(250):
        t = 10*i
        ntraj = len(ens.get_trajs(t))
        if wei.shape[0] != ntraj:
            print("clone")
            temp = np.zeros(ntraj, dtype=np.complex128)
            temp[:ntraj-1] = wei
            cln: Cloning = ens[0].clone[ntraj-1]
            temp[-1] = cln.trans * wei[cln.parent]
            temp[cln.parent] = np.sqrt(1 - cln.trans**2) * wei[cln.parent]
            wei = temp
        # print((2 * (1 - t) + 1j/2) * np.exp(-2*(1 - t)**2 - 1/2))
        print(t)
        wei = ens.propagate_weights(wei, t, t+10)
        print()
    breakpoint()

# TODO: vectorise
def ele_ovl(t1: View, t2: View):
    return np.eye(t1.nst)

def ele_ddt(t1: View, t2: View):
    return 1/2 * (t1.tdc + t2.tdc)

@cache
def nuc_ovl(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mpos = (t1.pos + t2.pos)/2
    dmom = t2.mom - t1.mom
    dphs = t2.phs - t1.phs
    res = -np.sum(t1.wid[:,None] * dpos**2)/2
    res -= np.sum(1/t1.wid[:,None] * dmom**2)/8
    res += 1j * np.sum(t1.pos * t1.mom - t2.pos * t2.mom + mpos * dmom)
    res += 1j * dphs
    return np.exp(res)

def nuc_pos(t1: View, t2: View):
    mpos = (t1.pos + t2.pos)/2
    dmom = t2.mom - t1.mom
    return (1j/t1.wid[:,None] * dmom/4 + mpos) * nuc_ovl(t1, t2)

def nuc_del(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mmom = (t1.mom + t2.mom)/2
    return (1j * mmom + t1.wid[:,None] * dpos) * nuc_ovl(t1, t2)

def nuc_del2(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mmom = (t1.mom + t2.mom)/2
    res = 2j * np.sum(t1.wid[:,None] * dpos * mmom)
    res -= 3 * np.sum(t1.wid)
    res += np.sum(t1.wid[:,None]**2 * dpos**2)
    res -= np.sum(mmom**2)
    res *= nuc_ovl(t1, t2)
    return res

# include masses more efficiently
def nuc_ke(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mmom = (t1.mom + t2.mom)/2
    res = 2j * np.sum(t1.wid[:,None] * dpos * mmom / t1.mass[:,None])
    res -= np.sum(t1.wid / t1.mass)
    res += np.sum(t1.wid[:,None]**2 * dpos**2 / t1.mass[:,None])
    res -= np.sum(mmom**2 / t1.mass[:,None])
    res *= -1/2 * nuc_ovl(t1, t2)
    return res

def nuc_ddr(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mmom = (t1.mom + t2.mom)/2
    res = -1j * mmom
    res -= t1.wid[:,None] * dpos
    return res * nuc_ovl(t1, t2)

def nuc_ddp(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    dmom = t2.mom - t1.mom
    res = -1j/2 * dpos
    res -= 1/(4 * t1.wid[:,None]) * dmom
    return res * nuc_ovl(t1, t2)

def nuc_ddt(t1: View, t2: View):
    res = np.sum(t2.vel * nuc_ddr(t1, t2))
    res += np.sum(t2.acc * t2.mass[:,None] * nuc_ddp(t1, t2))
    res += 1j * np.sum(t2.mom * t2.vel)/2 * nuc_ovl(t1, t2)
    return res

def tbf_ovl(t1: View, t2: View):
    return np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, ele_ovl(t1, t2)) * nuc_ovl(t1, t2)
    # return np.einsum("i,j->", np.conj(t1.amp), t2.amp) * nuc_ovl(t1, t2)

def tbf_ham(t1: View, t2: View):
    ovlp = nuc_ovl(t1, t2)
    kin = nuc_ke(t1, t2)
    pos = nuc_pos(t1, t2)
    pot = 1/2 * (t1.ham + t2.ham) * ovlp
    pot += 1/2 * np.diag(np.einsum("ad, sad -> s", pos, t1.grad + t2.grad))
    pot -= 1/2 * np.diag(np.einsum("ad, sad -> s", t1.pos, t1.grad) + np.einsum("ad, sad -> s", t2.pos, t2.grad)) * ovlp

    ham = np.eye(t1.nst) * kin
    ham += np.eye(t1.nst) * pot
    return np.einsum("i, j, ij ->", np.conj(t1.amp), t2.amp, ham * ele_ovl(t1, t2))

def tbf_ddt(t1: View, t2: View):
    amp_dot = -(1j * t2.ham + t2.tdc) @ t2.amp
    # res = np.einsum("i,j,ij->", np.conj(t1.amp), amp_dot, ele_ovl(t1, t2)) * nuc_ovl(t1, t2)
    res = np.einsum("i,j,ij->", np.conj(t1.amp), amp_dot, ele_ovl(t1, t2)) * nuc_ovl(t1, t2)
    res += np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, ele_ddt(t1, t2)) * nuc_ovl(t1, t2)
    res += np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, ele_ovl(t1, t2)) * nuc_ddt(t1, t2)
    return res

if __name__ == "__main__":
    main()