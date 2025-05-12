import numpy as np
from scipy.linalg import expm, fractional_matrix_power, solve
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import linear_sum_assignment, minimize_scalar
from functools import cache
import h5py
import os, sys
import time
import pickle
from classes.constants import convert, atomic_widths

class View:
    def __init__(self, data, step, act = None):
        if act is None:
            act = data.act[step]
        nst, nat, ndim = data.grad[0,0].shape
        ndof = nat*ndim
        self.mass = np.tile(data.mass[:,None], ndim).flatten()
        self.wid = np.tile(data.wid[:,None], ndim).flatten()

        self.pos = data.pos[step, act].reshape((-1, ndof))
        self.vel = data.vel[step, act].reshape((-1, ndof))
        self.acc = data.acc[step, act].reshape((-1, ndof))

        self.amp = data.amp[step, act]
        self.ham = data.ham[step, act]
        self.grad = data.grad[step, act].reshape((-1, nst, ndof))
        self.nac = data.nac[step, act].reshape((-1, nst, nst, ndof))
        np.nan_to_num(self.nac, copy=False)
        self.phs = data.phs[step, act]

        self.dpos =  self.pos[None,:] - self.pos[:,None]
        self.mpos = (self.pos[None,:] + self.pos[:,None])/2
        self.dmom =  self.mom[None,:] - self.mom[:,None]
        self.mmom = (self.mom[None,:] + self.mom[:,None])/2
        self.dphs =  self.phs[None,:] - self.phs[:,None]

        self.ovl = self.nuc_ovl()

    @property
    def mom(self):
        return self.vel * self.mass

    @property
    def tdc(self):
        return np.einsum("md, mijd -> mij", self.vel, self.nac)

    def nuc_ovl(self):
        res = 1j * self.dphs
        res -= np.sum(self.wid * self.dpos**2, axis=2)/2
        res -= np.sum(1/self.wid * self.dmom**2, axis=2)/8
        res += 1j * np.sum(self.pos*self.mom, axis=1)[:,None]
        res -= 1j * np.sum(self.pos*self.mom, axis=1)[None,:]
        res += 1j * np.sum(self.mpos*self.dmom, axis=2)
        return np.exp(res)

    def nuc_pos(self):
        return (1j/self.wid * self.dmom/4 + self.mpos) * self.ovl[:,:,None]

    def nuc_del(self):
        return (1j * self.mmom + self.wid * self.dpos) * self.ovl[:,:,None]

    def nuc_del2(self):
        res = 2j * np.sum(self.wid * self.dpos * self.mmom, axis=2)
        res -= np.sum(self.wid)
        res += np.sum(self.wid**2 * self.dpos**2, axis=2)
        res -= np.sum(self.mmom**2, axis=2)
        return res * self.ovl

    def nuc_ke(self):
        res = 2j * np.sum(self.wid * self.dpos * self.mmom / self.mass, axis=2)
        res -= np.sum(self.wid / self.mass) / 3
        res += np.sum(self.wid**2 * self.dpos**2 / self.mass, axis=2)
        res -= np.sum(self.mmom**2 / self.mass, axis=2)
        return -1/2 * res * self.ovl

    def nuc_ddr(self):
        res = -1j * self.mmom
        res -= self.wid * self.dpos
        return res * self.ovl[:,:,None]

    def nuc_ddp(self):
        res = -1j/2 * self.dpos
        res -= 1/self.wid * self.dmom / 4
        return res * self.ovl[:,:,None]

    def nuc_ddt(self):
        res = np.sum(self.vel[None,:] * self.nuc_ddr(), axis=2)
        res += np.sum(self.acc[None,:] * self.mass * self.nuc_ddp(), axis=2)
        res += 1j * np.sum(self.mom * self.vel, axis=1)[None,:]/2 * self.ovl
        return res

    def tbf_ovl(self):
        return np.einsum("mi,ni->mn", self.amp.conj(), self.amp) * self.ovl

    def tbf_ham(self):
        nst = self.grad.shape[1]
        kin = self.nuc_ke()
        pos = self.nuc_pos()
        pot = (self.ham[:,None] + self.ham[None,:]) * self.ovl[:,:,None,None] / 2
        pot += np.einsum("mnd,mid,ij->mnij", pos, self.grad, np.eye(nst)) / 2
        pot += np.einsum("mnd,nid,ij->mnij", pos, self.grad, np.eye(nst)) / 2
        temp = np.einsum("md,mid,ij->mij", self.pos, self.grad, np.eye(nst))
        pot -= (temp[:,None] + temp[None,:]) * self.ovl[:,:,None,None] / 2

        ham = np.einsum("ij,mn->mnij", np.eye(nst), kin)
        ham += np.einsum("ij,mnij->mnij", np.eye(nst), pot)
        ham += 1j * (self.tdc[:,None] + self.tdc[None,:]) * self.ovl[:,:,None,None] / 2
        return np.einsum("mi,nj,mnij->mn", self.amp.conj(), self.amp, ham)

    def tbf_ddt(self):
        res = np.einsum("mi,ni->mn", self.amp.conj(), self.amp) * self.nuc_ddt()
        amp_dot = -np.einsum("mij,mj->mi", 1j * self.ham + self.tdc, self.amp)
        res += np.einsum("mi,ni->mn", self.amp.conj(), amp_dot) * self.ovl
        return res

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

        self.ovl_old = None
        self.ham_old = None
        self.ddt_old = None
        self.hef_old = None
        self.vie_old = None

        self.set_clones()
        self.read_ensemble()
        self.save()

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
            # self.wid = np.array([atomic_widths[at] for at in atoms])
            self.wid = np.array([1e-33 for at in atoms])
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
                if not self.act[istep, itraj]:
                    continue

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

    @property
    def n_act(self):
        return np.sum(self.act[self.step])

    def setup(self):
        self.out = open("weighs.dat", "w")
        self.pops = open("pops.dat", "w")
        self.log = open("mce.log", "w")

        self.step = 0
        self.vie_old = self.get_views(0)
        self.ovl_old, self.ham_old, self.ddt_old = self.get_matrices(self.vie_old)
        self.hef_old = self.get_ham(self.ovl_old, self.ham_old - 1.j*self.ddt_old)

        n_act = np.sum(self.act[0])
        wei = np.ones(n_act) / n_act
        self.wei[0, self.act[0]] = wei / np.sqrt((wei.conj() @ self.ovl_old @ wei))

        self.trans = Transform(self.act[0], self.ovl_old)

    def check_clones(self):
        st = self.step
        rm = []
        cloned = False
        for clone in self.clones:
            if clone.step == st:
                cloned = True
                print("Clone")
                rm.append(clone)
                w1 = self.wei[st, clone.parent] * clone.trans
                w2 = self.wei[st, clone.parent] * np.sqrt(1 - clone.trans**2)
                self.wei[st, clone.child] = w1
                self.wei[st, clone.parent] = w2

        for r in rm:
            self.clones.remove(r)

        return cloned

    def get_views(self, step, act = None):
        if act is None:
            act = self.act[step]
        return View(self, step, act)

    def get_matrices(self, view: View):
        # ovl = np.zeros((self.n_act, self.n_act), dtype=np.complex128)
        # ddt = np.zeros((self.n_act, self.n_act), dtype=np.complex128)
        # ham = np.zeros((self.n_act, self.n_act), dtype=np.complex128)

        # for i, v1 in enumerate(views):
        #     for j, v2 in enumerate(views):
        #         ovl[i,j] = self.tbf_ovl(v1, v2)
        #         ddt[i,j] = self.tbf_ddt(v1, v2)
        #         ham[i,j] = self.tbf_ham(v1, v2)

        # return ovl, ham, ddt

        ovl = view.tbf_ovl()
        ham = view.tbf_ham()
        ddt = view.tbf_ddt()
        return ovl, ham, ddt

    def write_pops(self, time, wei, amp, ovl):
        self.pops.write(f"{time} ")
        pops = np.real(np.einsum('n,m,ns,ms,nm->s', wei.conj(), wei, amp.conj(), amp, ovl))
        # pops /= np.sum(pops)
        self.pops.write(" ".join(str(pop) for pop in pops))
        self.pops.write(f" {np.sum(pops)}")
        self.pops.write("\n")

    def write_coeffs(self, step):
        self.out.write(str(self.time[step]))
        for itraj in range(self.n_traj):
            self.out.write(f" {self.wei[step, itraj].real} {self.wei[step, itraj].imag}")
        self.out.write("\n")

    @staticmethod
    def _gcv(lam, A, B):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        filt = s**2 / (s**2 + lam)

        residuals = 0
        for i in range(B.shape[1]):
            bi = B[:, i]
            numerator = np.sum((filt * (U.T @ bi))**2)
            residuals += (np.linalg.norm(bi)**2 - numerator)

        denominator = np.sum(filt)**2
        return residuals / denominator

    @staticmethod
    def _opt_lam(A, B, lam_bounds=(1e-10, 1e-1)):
        result = minimize_scalar(
            MCE._gcv,
            bounds=lam_bounds,
            args=(A, B),
            method='bounded'
        )
        return result.x

    @staticmethod
    def _tikhonov(A, lam):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_inv = s / (s**2 + lam)
        return Vt.T @ np.diag(s_inv) @ U.T

    def get_ham(self, A, B):
        U, S, VT = np.linalg.svd(A)
        eps = S[0] * 1e-6
        S[S < eps] = eps
        Anew = U @ np.diag(S) @ VT
        for i in range(Anew.shape[0]):
            Anew[i,i] = 1
        return solve(Anew, B)
        lam = MCE._opt_lam(A, B)
        Ainv = MCE._tikhonov(A, lam)
        return Ainv @ B

    def run_step(self):
        st = self.step
        print(st)
        print(self.time[st])

        cloned = self.check_clones()

        wei = np.zeros((self.n_act), dtype=np.complex128)
        if cloned:
            vie_old = self.get_views(self.step)
            ovl_old, ham_old, ddt_old = self.get_matrices(vie_old)
            hef_old = self.get_ham(ovl_old, ham_old - 1.j*ddt_old)
        else:
            vie_old = self.vie_old
            ovl_old = self.ovl_old
            ham_old = self.ham_old
            ddt_old = self.ddt_old
            hef_old = self.hef_old

        t = time.time()
        vie_new = self.get_views(st + 1, self.act[st])
        print("View: ", time.time() - t, " s")

        t = time.time()
        ovl_new, ham_new, ddt_new = self.get_matrices(vie_new)
        print("Matrices: ", time.time() - t, " s")


        # if np.sum(self.act[st]) > 1:
            # breakpoint()

        print(ovl_old)
        t = time.time()
        hef_new = self.get_ham(ovl_new, ham_new - 1.j*ddt_new)
        print("Inverse: ", time.time() - t, " s")

        dt = self.time[st + 1] - self.time[st]
        wei = self.wei[st, self.act[st]]
        if cloned:
            wei = wei / np.sqrt((wei.conj() @ ovl_old @ wei))

        print(np.abs(wei)**2)
        print(f"Energy: {np.sum(np.outer(wei.conj(), wei) * ham_old)}")
        print("before: ", np.real(np.einsum("a, ab, b ->", wei.conj(), ovl_old, wei)))

        self.write_coeffs(self.step)
        self.write_pops(self.time[st], wei, vie_old.amp, ovl_old)


        # tr_ini, tr_fin = self.trans(self.act[st], ovl_ini, ovl_fin)
        # trH_ini = np.linalg.inv(tr_ini)
        # trH_fin = np.linalg.inv(tr_fin)
#
        # ddt_ini = trH_ini @ ddt_ini @ tr_ini
        # ham_ini = trH_ini @ ham_ini @ tr_ini
#
        # ddt_fin = trH_fin @ ddt_fin @ tr_fin
        # ham_fin = trH_fin @ ham_fin @ tr_fin

        # wei = trH_ini @ wei


        def linint(val0, val1, frac):
            return val0*(1-frac) + val1*frac

        t = time.time()
        n_substep = 20
        for i in range(n_substep):
            frac = (i + 0.5) / n_substep
            heff = linint(hef_old, hef_new, frac)
            # wei = expm_multiply(-1.j* heff * dt/n_substep, wei)
            wei = expm(-1.j* heff * dt/n_substep) @ wei
        print("Propagation: ", time.time() - t, " s")

        # wei = tr_fin @ wei
        if np.real(np.einsum("a, ab, b ->", wei.conj(), ovl_new, wei)) > 1.1:
            breakpoint()
        print("after:  ", np.real(np.einsum("a, ab, b ->", wei.conj(), ovl_new, wei)))
        wei = wei / np.sqrt((wei.conj() @ ovl_new @ wei))
        print(np.abs(wei)**2)
        print()

        i = 0
        for itraj in range(self.n_traj):
            if self.act[st, itraj]:
                self.wei[st + 1, itraj] = wei[i]
                i += 1

        self.step += 1

        self.vie_old = vie_new
        self.ovl_old = ovl_new
        self.ham_old = ham_new
        self.ddt_old = ddt_new
        self.hef_old = hef_new

    def finish(self):
        self.out.close()
        self.pops.close()
        self.log.close()

    def __call__(self):
        self.setup()
        for step in range(self.n_step - 1):
            self.run_step()
        self.finish()

    def save(self):
        with open("mce.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        with open(file, "rb") as f:
            return pickle.load(f)

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
    if len(sys.argv) < 2:
        mce = MCE.load("mce.pkl")
    else:
        mce = MCE(2000)
    mce()