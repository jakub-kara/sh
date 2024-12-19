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

class Cloning:
    def __init__(self, parent, child, transfer, step, time):
        self.parent = parent
        self.child = child
        self.trans = transfer
        self.time = time
        self.step = step

def get_dirs(loc=".", cond=lambda x : True):
    return [d for d in os.listdir(loc) if (os.path.isdir(d) and not d.startswith(".") and cond(d))]

def read_params(file: str):
    with h5py.File(file, "r") as f:
        atoms = f["info/ats"][:].astype("<U2")
        wid[:] = np.array([atomic_widths[at] for at in atoms])
        mass[:] = f["info/mass"][:]
        for step in range(n_step):
            time[step] = f[f"{step}/time"][()]

def read_ensemble():
    itraj = 0
    for dr in sorted(get_dirs()):
        os.chdir(dr)
        print(os.getcwd())
        itraj = read_bundle(itraj)
        os.chdir("..")

def read_bundle(itraj: int):
    for dr in sorted(get_dirs()):
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

            phs[step, itraj] = f[f"{step}/phase"][()]

def set_clones():
    itraj = 0
    for bund in get_dirs():
        os.chdir(bund)
        temp = 1
        with open("events.log", "r") as f:
            act[:, itraj] = True
            # clones.append(Cloning(itraj, itraj, 1, -1, -1))

            for line in f.readlines():
                if line.startswith("CLONE"):
                    temp += 1
                    data = line.split()
                    parent, child = itraj + int(data[1]), itraj + int(data[3])
                    transfer = float(data[4])
                    step, time = int(data[5]), float(data[6])
                    clones.append(Cloning(parent, child, transfer, step, time))
                    act[step:, child] = True
        itraj += temp
        os.chdir("..")

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
    ham += 1j/2*ovlp*(t1.tdc + t2.tdc)
    return np.einsum("i, j, ij ->", np.conj(t1.amp), t2.amp, ham * ele_ovl(t1, t2))

def tbf_ddt(t1: View, t2: View):
    amp_dot = -(1j * t2.ham + t2.tdc) @ t2.amp
    # res = np.einsum("i,j,ij->", np.conj(t1.amp), amp_dot, ele_ovl(t1, t2)) * nuc_ovl(t1, t2)
    res = np.einsum("i,j,ij->", np.conj(t1.amp), amp_dot, ele_ovl(t1, t2)) * nuc_ovl(t1, t2)
    # res += np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, ele_ddt(t1, t2)) * nuc_ovl(t1, t2)
    res += np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, ele_ovl(t1, t2)) * nuc_ddt(t1, t2)
    return res

n_step = 350
n_bund = 1
n_traj = 8
traj = 0
n_atoms = 1
n_states = 2
clones: list[Cloning] = []

time = np.zeros(n_step)
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
set_clones()
# act[:,0] = True
wei[0,0] = 1

# breakpoint()
for step in range(n_step-1):
    rm = []
    for c in clones:
        if c.step == step:
            rm.append(c)
            w1 = wei[step, c.parent] * c.trans
            w2 = wei[step, c.parent] * np.sqrt(1 - c.trans**2)
            v1 = View(step, c.child)
            v2 = View(step, c.parent)
            ovl = tbf_ovl(v1, v2)
            tot = np.abs(w1)**2 + np.abs(w2)**2 + 2*np.real(np.conj(w1) * w2) * ovl
            w1 /= np.sqrt(tot)
            w2 /= np.sqrt(tot)
            wei[step, c.child] = w1
            wei[step, c.parent] = w2

    for r in rm:
        clones.remove(r)

    n_act = np.sum(act[step])
    wei_fin = np.zeros((n_act), dtype=np.complex128)
    ovl_ini = np.zeros((n_act, n_act), dtype=np.complex128)
    ovl_fin = np.zeros((n_act, n_act), dtype=np.complex128)
    ovl_mat = np.zeros((n_act, n_act), dtype=np.complex128)
    ddt_ini = np.zeros((n_act, n_act), dtype=np.complex128)
    ddt_fin = np.zeros((n_act, n_act), dtype=np.complex128)
    ham_ini = np.zeros((n_act, n_act), dtype=np.complex128)
    ham_fin = np.zeros((n_act, n_act), dtype=np.complex128)

    view_ini = np.empty(n_act, dtype=object)
    view_fin = np.empty(n_act, dtype=object)

    i = 0
    for itraj in range(n_traj):
        if act[step, itraj]:
            view_ini[i] = View(step, itraj)
            view_fin[i] = View(step + 1, itraj)
            i += 1

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

    for i, v1 in enumerate(view_ini):
        for j, v2 in enumerate(view_fin):
            ovl_mat[i,j] = tbf_ovl(v1, v2)

    ddt_ini -= ddt_ini.T.conj() * (1 - np.eye(n_act))
    ddt_ini *= (1 - np.eye(n_act)) / 2 + np.eye(n_act)
    ddt_fin -= ddt_fin.T.conj() * (1 - np.eye(n_act))
    ddt_fin *= (1 - np.eye(n_act)) / 2 + np.eye(n_act)

    # val, vec = np.linalg.eigh(ovl_ini)
    # tr = vec @ np.sqrt(np.diag(1 / val)) @ vec.T
    # trH = tr.T.conj()

    # ovl_ini = np.eye(n_act)
    # ddt_ini = tr @ ddt_ini @ trH
    # ham_ini = tr @ ham_ini @ trH
    # ovl_fin = tr @ ovl_fin @ trH
    # ddt_fin = tr @ ddt_fin @ trH
    # ham_fin = tr @ ham_fin @ trH

    dt = time[step + 1] - time[step]
    wei_fin = wei[step, act[step]]
    # wei_fin = tr @ wei_fin

    if n_act > 1:
        breakpoint()

    def hamint(frac):
        return (1 - frac) * ham_ini + frac * ham_fin

    def ovlint(frac):
        return (1 - frac) * np.linalg.inv(ovl_ini) + frac * np.linalg.inv(ovl_fin)

    def ddtint(frac):
        return (1 - frac) * ddt_ini + frac * ddt_fin

    # print("before: ", np.sum(np.abs(wei_fin)**2))
    print("before: ", np.conj(wei_fin) @ ovl_ini @ wei_fin)

    prop = np.eye(n_act)
    n_substep = 20
    h0 = ham_ini
    htra = ovl_mat @ ham_fin @ ovl_mat.T.conj()
    for i in range(n_substep):
        hk = h0 + i / n_substep * (htra - h0)
        rk = expm(-1j * hk * dt / n_substep)
        prop = rk @ prop
    prop = ovl_mat.T.conj() @ prop
    wei_fin = prop @ wei_fin

    # for i in range(n_substep):
    #     frac = (i + 0.5) / n_substep
    #     wei_fin = expm(ovlint(frac) @ (-1j*hamint(frac) + ddtint(frac)) * dt/n_substep) @ wei_fin

    print("after:  ", np.conj(wei_fin) @ ovl_fin @ wei_fin)
    # print("after:  ", np.sum(np.abs(wei_fin)**2))
    print()
    # wei_fin = trH @ wei_fin

    i = 0
    for itraj in range(n_traj):
        if act[step, itraj]:
            wei[step + 1, itraj] = wei_fin[i]
            i += 1



breakpoint()