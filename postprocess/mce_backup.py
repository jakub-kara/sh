import numpy as np
from scipy.linalg import expm, fractional_matrix_power
from scipy.optimize import linear_sum_assignment
from functools import cache
import h5py
import os, sys
from classes.constants import convert, atomic_widths

class View:
    def __init__(self, data, step, itraj):
        self.mass = data.mass

        self.pos = data.pos[step, itraj]
        self.vel = data.vel[step, itraj]
        self.acc = data.acc[step, itraj]

        self.amp = data.amp[step, itraj]
        self.ham = data.ham[step, itraj]
        self.grad = data.grad[step, itraj]
        self.nac = data.nac[step, itraj]
        np.nan_to_num(self.nac, copy=False)
        self.phs = data.phs[step, itraj]

    @property
    def mom(self):
        return self.vel * self.mass[:,None]

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

class MCE:
    def __init__(self, n_step):
        self.n_step = n_step
        self.bunds = sys.argv[1:]
        self.get_ntraj()
        self.read_params()

        self.clones: list[Cloning] = []
        self.out = open("weighs.dat", "w")
        self.pops = open("pops.dat", "w")
        self.log = open("ref.log", "w")

        n_traj = self.n_traj
        n_atoms = self.n_atoms
        n_states = self.n_states
        self.pos = np.zeros((n_step, n_traj, n_atoms, 3))
        self.vel = np.zeros((n_step, n_traj, n_atoms, 3))
        self.acc = np.zeros((n_step, n_traj, n_atoms, 3))
        self.amp = np.zeros((n_step, n_traj, n_states), dtype=np.complex128)
        self.ham = np.zeros((n_step, n_traj, n_states, n_states))
        self.grad = np.zeros((n_step, n_traj, n_states, n_atoms, 3))
        self.nac = np.zeros((n_step, n_traj, n_states, n_states, n_atoms, 3))
        self.tdc = np.zeros((n_step, n_traj, n_states, n_states))
        self.phs = np.zeros((n_step, n_traj))

        self.wei = np.zeros((n_step, n_traj), dtype=np.complex128)
        self.act = np.zeros((n_step, n_traj), dtype=bool)

        self.ovl_phs = np.ones(n_traj)

        self.read_ensemble()
        self.set_clones()

    @property
    def n_bunds(self):
        return len(self.bunds)

    @property
    def n_atoms(self):
        return len(self.mass)

    @property
    def n_steps(self):
        return len(self.time)

    def get_ntraj(self):
        self.n_traj = 0
        orig = os.getcwd()
        for bund in self.bunds:
            os.chdir(bund)
            self.n_traj += len(get_dirs())
            os.chdir(orig)

    def read_params(self):
        with h5py.File(f"{self.bunds[0]}/0/data/out.h5", "r") as f:
            self.n_states = f["info/nst"][()]

            atoms = f["info/ats"][:].astype("<U2")
            self.wid = np.array([atomic_widths[at] for at in atoms])
            self.mass = f["info/mass"][:]

            time = []
            for step in f.keys():
                if step == "info":
                    continue
                time.append(f[f"{step}/time"][()])
        time = np.array(time)
        time.sort()
        self.time = time

    def read_ensemble(self):
        self.itraj = 0
        orig = os.getcwd()
        for dr in self.bunds:
            os.chdir(dr)
            print(os.getcwd())
            self.read_bundle()
            os.chdir(orig)

    def read_bundle(self):
        for dr in sorted(get_dirs()):
            os.chdir(dr)
            print(f"  {os.getcwd()}, {self.itraj}")
            self.read_traj("data/out.h5")
            self.itraj += 1
            os.chdir("..")

    def read_traj(self, file: str):
        with h5py.File(file, "r") as f:
            keys = [key for key in f.keys()]
            keys.remove("info")
            steps = sorted(map(int, keys))
            for istep, step in enumerate(steps):
                if istep > self.n_step - 1:
                    return

                itraj = self.itraj
                self.pos[step, itraj] = f[f"{step}/pos"][:]
                self.vel[step, itraj] = f[f"{step}/vel"][:]
                self.acc[step, itraj] = f[f"{step}/acc"][:]

                self.amp[step, itraj] = f[f"{step}/coeff"][:]
                self.ham[step, itraj] = f[f"{step}/hdiag"][:]
                self.grad[step, itraj] = f[f"{step}/grad"][:]
                self.nac[step, itraj] = f[f"{step}/nacdr"][:]
                self.tdc[step, itraj] = f[f"{step}/nacdt"][:]

                self.phs[step, itraj] = f[f"{step}/phase"][()]

    def set_clones(self):
        itraj = 0
        orig = os.getcwd()
        for bund in self.bunds:
            os.chdir(bund)
            temp = 1
            with open("events.log", "r") as f:
                self.act[:, itraj] = True
                # clones.append(Cloning(itraj, itraj, 1, -1, -1))

                for line in f.readlines():
                    if line.startswith("CLONE"):
                        temp += 1
                        data = line.split()
                        parent, child = itraj + int(data[1]), itraj + int(data[3])
                        transfer = float(data[4])
                        step, time = int(data[5]), float(data[6])
                        self.clones.append(Cloning(parent, child, transfer, step, time))
                        self.act[step:, child] = True
            itraj += temp
            os.chdir(orig)

    # TODO: vectorise
    def ele_ovl(self, t1: View, t2: View):
        return np.eye(self.n_states)

    def ele_ddt(self, t1: View, t2: View):
        return 1/2 * (t1.tdc + t2.tdc)

    @cache
    def nuc_ovl(self, t1: View, t2: View):
        dpos = t2.pos - t1.pos
        mpos = (t1.pos + t2.pos)/2
        dmom = t2.mom - t1.mom
        dphs = t2.phs - t1.phs
        res = -np.sum(self.wid[:,None] * dpos**2)/2
        res -= np.sum(1/self.wid[:,None] * dmom**2)/8
        res += 1j * np.sum(t1.pos * t1.mom - t2.pos * t2.mom + mpos * dmom)
        res += 1j * dphs
        return np.exp(res)

    def nuc_pos(self, t1: View, t2: View):
        mpos = (t1.pos + t2.pos)/2
        dmom = t2.mom - t1.mom
        return (1j/self.wid[:,None] * dmom/4 + mpos) * self.nuc_ovl(t1, t2)

    def nuc_del(self, t1: View, t2: View):
        dpos = t2.pos - t1.pos
        mmom = (t1.mom + t2.mom)/2
        return (1j * mmom + self.wid[:,None] * dpos) * self.nuc_ovl(t1, t2)

    def nuc_del2(self, t1: View, t2: View):
        dpos = t2.pos - t1.pos
        mmom = (t1.mom + t2.mom)/2
        res = 2j * np.sum(self.wid[:,None] * dpos * mmom)
        res -= 3 * np.sum(self.wid)
        res += np.sum(self.wid[:,None]**2 * dpos**2)
        res -= np.sum(mmom**2)
        res *= self.nuc_ovl(t1, t2)
        return res

    # include masses more efficiently
    def nuc_ke(self, t1: View, t2: View):
        dpos = t2.pos - t1.pos
        mmom = (t1.mom + t2.mom)/2
        res = 2j * np.sum(self.wid[:,None] * dpos * mmom / self.mass[:,None])
        res -= np.sum(self.wid / self.mass)
        res += np.sum(self.wid[:,None]**2 * dpos**2 / self.mass[:,None])
        res -= np.sum(mmom**2 / self.mass[:,None])
        res *= -1/2 * self.nuc_ovl(t1, t2)
        return res

    def nuc_ddr(self, t1: View, t2: View):
        dpos = t2.pos - t1.pos
        mmom = (t1.mom + t2.mom)/2
        res = -1j * mmom
        res -= self.wid[:,None] * dpos
        return res * self.nuc_ovl(t1, t2)

    def nuc_ddp(self, t1: View, t2: View):
        dpos = t2.pos - t1.pos
        dmom = t2.mom - t1.mom
        res = -1j/2 * dpos
        res -= 1/(4 * self.wid[:,None]) * dmom
        return res * self.nuc_ovl(t1, t2)

    def nuc_ddt(self, t1: View, t2: View):
        res = np.sum(t2.vel * self.nuc_ddr(t1, t2))
        res += np.sum(t2.acc * self.mass[:,None] * self.nuc_ddp(t1, t2))
        res += 1j * np.sum(t2.mom * t2.vel)/2 * self.nuc_ovl(t1, t2)
        return res

    def tbf_ovl(self, t1: View, t2: View):
        return np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, self.ele_ovl(t1, t2)) * self.nuc_ovl(t1, t2)
        # return np.einsum("i,j->", np.conj(t1.amp), t2.amp) * nuc_ovl(t1, t2)

    def tbf_ham(self, t1: View, t2: View):
        ovlp = self.nuc_ovl(t1, t2)
        kin = self.nuc_ke(t1, t2)
        pos = self.nuc_pos(t1, t2)
        pot = 1/2 * (t1.ham + t2.ham) * ovlp
        pot += 1/2 * np.diag(np.einsum("ad, sad -> s", pos, t1.grad + t2.grad))
        pot -= 1/2 * np.diag(np.einsum("ad, sad -> s", t1.pos, t1.grad) + np.einsum("ad, sad -> s", t2.pos, t2.grad)) * ovlp

        ham = np.eye(self.n_states) * kin
        ham += np.eye(self.n_states) * pot
        ham += 1j/2*ovlp*(t1.tdc + t2.tdc)
        return np.einsum("i, j, ij ->", np.conj(t1.amp), t2.amp, ham)

    def tbf_ddt(self, t1: View, t2: View):
        # if t1 is not t2:
        #     return 0
        amp_dot = -(1j * t2.ham + t2.tdc) @ t2.amp
        # res = np.einsum("i,j,ij->", np.conj(t1.amp), amp_dot, ele_ovl(t1, t2)) * nuc_ovl(t1, t2)
        res = np.einsum("i,j,ij->", np.conj(t1.amp), amp_dot, self.ele_ovl(t1, t2)) * self.nuc_ovl(t1, t2)
        # res += np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, ele_ddt(t1, t2)) * nuc_ovl(t1, t2)
        res += np.einsum("i,j,ij->", np.conj(t1.amp), t2.amp, self.ele_ovl(t1, t2)) * self.nuc_ddt(t1, t2)
        return res

    @property
    def n_act(self):
        return np.sum(self.act[self.step])

    def setup(self):
        self.step = 0
        n_act = np.sum(self.act[0])
        wei_ini = np.ones(n_act) / n_act
        view_ini = np.empty(n_act, dtype=object)

        i = 0
        for itraj in range(self.n_traj):
            if self.act[0, itraj]:
                view_ini[i] = View(self, 0, itraj)
                i += 1

        ovl_ini = np.zeros((n_act, n_act), dtype=np.complex128)
        for i, v1 in enumerate(view_ini):
            for j, v2 in enumerate(view_ini):
                ovl_ini[i,j] = self.tbf_ovl(v1, v2)
        self.wei[0, self.act[0]] = wei_ini / np.sqrt((wei_ini.conj() @ ovl_ini @ wei_ini))

        self.trans = Transform(self.act[0], ovl_ini)

    def check_clones(self):
        st = self.step
        rm = []
        for clone in self.clones:
            if clone.step == st:
                self.cloned = True
                print("Clone")
                rm.append(clone)
                w1 = self.wei[st, clone.parent] * clone.trans
                w2 = self.wei[st, clone.parent] * np.sqrt(1 - clone.trans**2)
                v1 = View(self, st, clone.child)
                v2 = View(self, st, clone.parent)
                ovl = self.tbf_ovl(v1, v2)
                # tot = np.abs(w1)**2 + np.abs(w2)**2 + 2*np.real(np.conj(w1) * w2) * ovl
                # w1 /= np.sqrt(tot)
                # w2 /= np.sqrt(tot)
                self.wei[st, clone.child] = w1
                self.wei[st, clone.parent] = w2

        for r in rm:
            self.clones.remove(r)

    def run_step(self):
        st = self.step
        print(st)
        print(self.time[st])

        self.cloned = False
        self.check_clones()

        wei_fin = np.zeros((self.n_act), dtype=np.complex128)
        ovl_ini = np.zeros((self.n_act, self.n_act), dtype=np.complex128)
        ovl_fin = np.zeros((self.n_act, self.n_act), dtype=np.complex128)
        ovl_mat = np.zeros((self.n_act, self.n_act), dtype=np.complex128)
        ddt_ini = np.zeros((self.n_act, self.n_act), dtype=np.complex128)
        ddt_fin = np.zeros((self.n_act, self.n_act), dtype=np.complex128)
        ham_ini = np.zeros((self.n_act, self.n_act), dtype=np.complex128)
        ham_fin = np.zeros((self.n_act, self.n_act), dtype=np.complex128)
        amp_ini = np.zeros((self.n_act, self.n_states), dtype=np.complex128)

        view_ini = np.empty(self.n_act, dtype=object)
        view_fin = np.empty(self.n_act, dtype=object)

        i = 0
        for itraj in range(self.n_traj):
            if self.act[st, itraj]:
                view_ini[i] = View(self, st, itraj)
                view_fin[i] = View(self, st + 1, itraj)
                i += 1

        for i, v1 in enumerate(view_ini):
            amp_ini[i] = v1.amp
            for j, v2 in enumerate(view_ini):
                ovl_ini[i,j] = self.tbf_ovl(v1, v2)
                ddt_ini[i,j] = self.tbf_ddt(v1, v2)
                ham_ini[i,j] = self.tbf_ham(v1, v2)

        for i, v1 in enumerate(view_fin):
            for j, v2 in enumerate(view_fin):
                ovl_fin[i,j] = self.tbf_ovl(v1, v2)
                ddt_fin[i,j] = self.tbf_ddt(v1, v2)
                ham_fin[i,j] = self.tbf_ham(v1, v2)

        for i, v1 in enumerate(view_ini):
            for j, v2 in enumerate(view_fin):
                ovl_mat[i,j] = self.tbf_ovl(v1, v2)

        if self.n_act > 1:
            self.log.write(f"{self.tbf_ham(view_ini[0], view_ini[1])}\n")

        dt = self.time[st + 1] - self.time[st]
        wei_fin = self.wei[st, self.act[st]]
        if self.cloned:
            wei_fin = wei_fin / np.sqrt((wei_fin.conj() @ ovl_ini @ wei_fin))



        print(np.abs(wei_fin)**2)
        print(f"Energy: {np.sum(np.outer(wei_fin.conj(), wei_fin) * ham_ini)}")
        print("before: ", np.real(np.einsum("a, ab, b ->", wei_fin.conj(), ovl_ini, wei_fin)))
        self.pops.write(f"{self.time[st]}")
        for state in range(self.n_states):
            self.pops.write(f" {np.real(np.einsum('n,m,n,m,nm->', wei_fin.conj(), wei_fin, amp_ini.conj()[:,state], amp_ini[:,state], ovl_ini))}")
        self.pops.write("\n")

        # tr_ini, tr_fin = self.trans(self.act[st], ovl_ini, ovl_fin)
        # trH_ini = np.linalg.inv(tr_ini)
        # trH_fin = np.linalg.inv(tr_fin)
#
        # ddt_ini = trH_ini @ ddt_ini @ tr_ini
        # ham_ini = trH_ini @ ham_ini @ tr_ini
#
        # ddt_fin = trH_fin @ ddt_fin @ tr_fin
        # ham_fin = trH_fin @ ham_fin @ tr_fin
#
        # ddt_ini -= ddt_ini.T.conj()
        # ddt_ini /= 2
        # ddt_fin -= ddt_fin.T.conj()
        # ddt_fin /= 2
        # ham_ini += ham_ini.T.conj() - ham_ini * np.eye(self.n_act)
        # ham_fin += ham_fin.T.conj() - ham_fin * np.eye(self.n_act)
#
        # wei_fin = trH_ini @ wei_fin


        def linint(val0, val1, frac):
            return val0*(1-frac) + val1*frac


        n_substep = 20
        heff_ini = np.linalg.inv(ovl_ini) @ (ham_ini - 1.j*ddt_ini)
        heff_fin = np.linalg.inv(ovl_fin) @ (ham_fin - 1.j*ddt_fin)

        for i in range(n_substep+1):
            frac = i / n_substep
            heff = linint(heff_ini, heff_fin, frac)
            wei_fin = expm(-1.j* heff * dt/n_substep) @ wei_fin

        # wei_fin = tr_fin @ wei_fin
        print("after:  ", np.real(np.einsum("a, ab, b ->", wei_fin.conj(), ovl_fin, wei_fin)))
        print(np.abs(wei_fin)**2)
        print()

        i = 0
        self.out.write(str(self.time[st]))
        for itraj in range(self.n_traj):
            if self.act[st, itraj]:
                self.wei[st + 1, itraj] = wei_fin[i]
                i += 1

            self.out.write(f" {self.wei[st, itraj].real} {self.wei[st, itraj].imag}")
        self.out.write("\n")

        self.step += 1

    def finish(self):
        self.out.close()
        self.pops.close()
        self.log.close()

    def __call__(self):
        self.setup()
        for step in range(self.n_step - 1):
            self.run_step()
        self.finish()

class Transform:
    def __init__(self, act, ovl_ini):
        self.act = act
        self.tr_ini = self.lowdin(ovl_ini)

    def __call__(self, act, ovl_ini, ovl_fin):
        nini = self.tr_ini.shape[0]
        tr_fin = self.lowdin(ovl_fin)
        nfin = tr_fin.shape[0]

        if nini < nfin:
            tr_exp = self.expand(self.tr_ini, nfin)
            tr_ini = self.lowdin(ovl_ini)
            tr_ini = self.permute(tr_ini, tr_exp)
            tr_ini = self.procrustes(tr_ini, tr_exp)
            tr_ini = self.phase(tr_ini, tr_exp)
        else:
            tr_ini = self.tr_ini.copy()


        breakpoint()
        # tr_fin = self.permute(tr_fin, tr_ini)
        # tr_fin = self.procrustes(tr_fin, tr_ini)
        # tr_fin = self.phase(tr_fin, tr_ini)
        print(tr_ini.T.conj() @ ovl_ini @ tr_ini)
        print(tr_fin.T.conj() @ ovl_fin @ tr_fin)

        print(tr_ini.T.conj() @ tr_fin)
        print(tr_ini)
        print(tr_fin)
        self.tr_ini = tr_fin
        return tr_ini, tr_fin

    def lowdin(self, ovl_mat):
        val, vec = np.linalg.eigh(ovl_mat)
        vec = fractional_matrix_power(ovl_mat, -0.5).T.conj() @ vec
        vec /= np.linalg.norm(vec, axis=0, keepdims=True)
        return vec

    def permute(self, mat, ref):
        ovl = np.abs(ref.T.conj() @ mat)**2
        row, col = linear_sum_assignment(-ovl)
        print(col)
        return mat[:,col]

    def procrustes(self, mat, ref):
        n = mat.shape[0]
        uu, _, vh = np.linalg.svd(ref.T.conj() @ mat)
        rot = uu @ vh
        rot[:,-1] /= np.linalg.det(rot)
        return mat @ rot.T.conj()

    def phase(self, mat, ref):
        for i in range(mat.shape[1]):
            phs = np.vdot(ref[:,i], mat[:,i])
            ang = np.angle(phs)
            mat[:,i] *= np.exp(-1.j*ang)
        return mat

    def expand(self, mat, dim):
        n = mat.shape[0]
        if n == dim:
            return mat
        pad = np.eye(dim, dtype=mat.dtype)
        pad[:n, :n] = mat
        return pad

if __name__ == "__main__":
    np.set_printoptions(linewidth=150, precision=8)
    mce = MCE(300)
    mce()