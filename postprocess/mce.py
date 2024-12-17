import numpy as np
from scipy.linalg import expm
from functools import cache
from bisect import bisect_right
import h5py
import os
from classes.constants import convert, atomic_widths

class View:
    def __init__(self, step, itraj):
        self.pos = pos[step, itraj]
        self.vel = vel[step, itraj]
        self.acc = acc[step, itraj]

        self.amp = amp[step, itraj]
        self.ham = ham[step, itraj]
        self.grad = grad[step, itraj]
        self.nac = nac[step, itraj]
        self.phs = phs[step, itraj]

    @property
    def mom(self):
        return self.vel * mass[:,None]

    @property
    def tdc(self):
        return np.einsum("ad, ijad -> ij", self.vel, self.nac)

def get_dirs(loc=".", cond=lambda x : True):
    return [d for d in os.listdir(loc) if (os.path.isdir(d) and not d.startswith(".") and cond(d))]

def read_params(file: str):
    with h5py.File(file, "r") as f:
        atoms = f["info/ats"][:].astype("<U2")
        wid[:] = np.array([atomic_widths[at] for at in atoms])
        mass[:] = f["info/mass"][:]

def read_ensemble():
    itraj = 0
    for dr in get_dirs():
        os.chdir(dr)
        print(os.getcwd())
        itraj = read_bundle(itraj)
        os.chdir("..")

def read_bundle(itraj: int):
    for dr in get_dirs():
        os.chdir(dr)
        print(f"  {os.getcwd()}, {itraj}")
        read_traj("data/out.h5", itraj)
        itraj += 1
        os.chdir("..")
    return itraj

def read_traj(file: str, itraj: int):
    with h5py.File(file, "r") as f:
        keys = [key for key in f.keys()]
        keys.remove("info")
        steps = sorted(map(int, keys))
        for istep, step in enumerate(steps):
            if istep > n_step - 1:
                return

            pos[step, itraj] = f[f"{step}/pos"][:]
            vel[step, itraj] = f[f"{step}/vel"][:]
            acc[step, itraj] = f[f"{step}/acc"][:]

            amp[step, itraj] = f[f"{step}/coeff"][:]
            ham[step, itraj] = f[f"{step}/hdiag"][:]
            grad[step, itraj] = f[f"{step}/grad"][:]
            nac[step, itraj] = f[f"{step}/nacdr"][:]
            tdc[step, itraj] = f[f"{step}/nacdt"][:]

            # phs[step, itraj] = f[f"{step}/phase"][()]

# TODO: vectorise
def ele_ovl(t1: View, t2: View):
    return np.eye(n_states)

def ele_ddt(t1: View, t2: View):
    return 1/2 * (t1.tdc + t2.tdc)

@cache
def nuc_ovl(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mpos = (t1.pos + t2.pos)/2
    dmom = t2.mom - t1.mom
    dphs = t2.phs - t1.phs
    res = -np.sum(wid[:,None] * dpos**2)/2
    res -= np.sum(1/wid[:,None] * dmom**2)/8
    res += 1j * np.sum(t1.pos * t1.mom - t2.pos * t2.mom + mpos * dmom)
    res += 1j * dphs
    return np.exp(res)

def nuc_pos(t1: View, t2: View):
    mpos = (t1.pos + t2.pos)/2
    dmom = t2.mom - t1.mom
    return (1j/wid[:,None] * dmom/4 + mpos) * nuc_ovl(t1, t2)

def nuc_del(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mmom = (t1.mom + t2.mom)/2
    return (1j * mmom + wid[:,None] * dpos) * nuc_ovl(t1, t2)

def nuc_del2(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mmom = (t1.mom + t2.mom)/2
    res = 2j * np.sum(wid[:,None] * dpos * mmom)
    res -= 3 * np.sum(wid)
    res += np.sum(wid[:,None]**2 * dpos**2)
    res -= np.sum(mmom**2)
    res *= nuc_ovl(t1, t2)
    return res

# include masses more efficiently
def nuc_ke(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mmom = (t1.mom + t2.mom)/2
    res = 2j * np.sum(wid[:,None] * dpos * mmom / mass[:,None])
    res -= np.sum(wid / mass)
    res += np.sum(wid[:,None]**2 * dpos**2 / mass[:,None])
    res -= np.sum(mmom**2 / mass[:,None])
    res *= -1/2 * nuc_ovl(t1, t2)
    return res

def nuc_ddr(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    mmom = (t1.mom + t2.mom)/2
    res = -1j * mmom
    res -= wid[:,None] * dpos
    return res * nuc_ovl(t1, t2)

def nuc_ddp(t1: View, t2: View):
    dpos = t2.pos - t1.pos
    dmom = t2.mom - t1.mom
    res = -1j/2 * dpos
    res -= 1/(4 * wid[:,None]) * dmom
    return res * nuc_ovl(t1, t2)

def nuc_ddt(t1: View, t2: View):
    res = np.sum(t2.vel * nuc_ddr(t1, t2))
    res += np.sum(t2.acc * mass[:,None] * nuc_ddp(t1, t2))
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

    ham = np.eye(n_states) * kin
    ham += np.eye(n_states) * pot
    return np.einsum("i, j, ij ->", np.conj(t1.amp), t2.amp, ham * ele_ovl(t1, t2))

def tbf_ddt(t1: View, t2: View):
    amp_dot = -(1j * t2.ham + t2.tdc) @ t2.amp
    # res = np.einsum("i,j,ij->", np.conj(t1.amp), amp_dot, ele_ovl(t1, t2)) * nuc_ovl(t1, t2)
    res = np.einsum("i,j,ij->", np.conj(t1.amp), amp_dot, ele_ovl(t1, t2)) * nuc_ovl(t1, t2)
    res += np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, ele_ddt(t1, t2)) * nuc_ovl(t1, t2)
    res += np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, ele_ovl(t1, t2)) * nuc_ddt(t1, t2)
    return res

n_step = 350
n_bund = 1
n_traj = 8
traj = 0
n_atoms = 1
n_states = 2

wid = np.zeros(n_atoms)
mass = np.zeros(n_atoms)
pos = np.zeros((n_step, n_traj, n_atoms, 3))
vel = np.zeros((n_step, n_traj, n_atoms, 3))
acc = np.zeros((n_step, n_traj, n_atoms, 3))
amp = np.zeros((n_step, n_traj, n_states), dtype=np.complex128)
ham = np.zeros((n_step, n_traj, n_states, n_states))
grad = np.zeros((n_step, n_traj, n_states, n_atoms, 3))
nac = np.zeros((n_step, n_traj, n_states, n_states, n_atoms, 3))
tdc = np.zeros((n_step, n_traj, n_states, n_states))
phs = np.zeros((n_step, n_traj))

wei = np.zeros((n_step, n_traj), dtype=np.complex128)
act = np.zeros((n_step, n_traj), dtype=bool)

read_params("T0/0/data/out.h5")
read_ensemble()


breakpoint()