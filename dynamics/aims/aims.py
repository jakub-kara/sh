import numpy as np
from scipy.linalg import expm
from copy import deepcopy
import shutil
import os
from classes.bundle import Bundle
from classes.constants import atomic_widths
from classes.config import Config, get_cfg
from classes.format import *
from classes.molecule import Molecule, MoleculeMixin
from classes.out import FileManager as fm
from classes.timestep import Timestep
from classes.trajectory import Trajectory
from dynamics.base import Dynamics
from electronic.base import ESTMode
from updaters.tdc import TDCUpdater


@Config.Option("aims", "method")
@Config("csthresh", default=0.01, path=Config.DYNAMICS, valid=Config.POSITIVE(Config.FLOAT))
@Config("poptospawn", default=0.01, path=Config.DYNAMICS, valid=Config.POSITIVE(Config.FLOAT))
@Config("omax", default=0.5, path=Config.DYNAMICS, valid=Config.POSITIVE(Config.FLOAT))
@Config("olapthresh", default=1e-3, path=Config.DYNAMICS, valid=Config.POSITIVE(Config.FLOAT))
@Config("regthresh", default=1e-4, path=Config.DYNAMICS, valid=Config.POSITIVE(Config.FLOAT))
class AIMS(Dynamics):
    mode = ESTMode(ESTMode.ACTIVE)

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)
        config["nuclear"]["mixins"] = "aims"

        self._csthresh = get_cfg(self, "csthresh", dynamics)
        self._poptospawn = get_cfg(self, "poptospawn", dynamics)
        self._omax = get_cfg(self, "omax", dynamics)
        self._olapthresh = get_cfg(self, "olapthresh", dynamics)
        self._regthresh = get_cfg(self, "regthresh", dynamics)

        self._clone = None
        self._spawnprop = None
        self._backprop = None
        self._spawnstate = None
        self._tspawn = None
        self._tentry = None

    def step_mode(self, mol):
        return self.mode(mol) + TDCUpdater().mode(mol)

    def calculate_acceleration(self, mol: Molecule):
        mol.acc_ad = -mol.grad_sad[mol.state] / mol.mass_a[:,None]

    def potential_energy(self, mol: Molecule):
        return mol.ham_eig_ss[mol.state, mol.state]

    def step_bundle(self, bundle: Bundle):
        self._set_active(bundle)

        # if bundle.n_traj > 1:
        #     breakpoint()

        print()
        if bundle.n_traj > 1:
            print(bundle.iactive, bundle.n_traj)

        os.chdir(f"{bundle.iactive}")
        self.step_traj(bundle.active)
        os.chdir("..")

        if self._spawnprop is None and self._backprop is None:
            print("NORMAL")
            temp = [traj.timestep.time for traj in bundle.trajs]
            if min(temp) == max(temp):
                print("UPDATING")
                self.update_amps(
                    [traj.mols[-2] for traj in bundle.trajs],
                    [traj.mols[-1] for traj in bundle.trajs],
                    bundle.active.timestep)

            self._check_spawning(bundle.active)

        if self._spawnstate is not None:
            print("SPAWNING")
            self._enter_spawning(bundle.active)

        if self._backprop:
            print("BACKPROP")
            if self._check_backprop(bundle.active):
                self._exit_backprop()
                bundle

        if self._clone:
            shutil.copytree(f"{bundle.iactive}", f"{bundle.n_traj}", dirs_exist_ok=True)
            with open("events.log", "a") as f:
                f.write(f"CLONE {bundle.iactive} {bundle.n_traj} {bundle.active.timestep.step} {bundle.active.timestep.time:.4f}\n")

            os.chdir(f"{bundle.n_traj}")
            self._clone.next_step()
            self._clone.write_outputs()
            os.chdir("..")
            bundle.add_trajectory(self._clone)
            self._clone = None

        if self._success:
            self.step_success(bundle)

    def _set_active(self, bundle: Bundle):
        if self._spawnprop is not None:
            bundle.active = self._spawnprop
        elif self._backprop is not None:
            bundle.active = self._backprop
        else:
            bundle.set_active()

        bundle.iactive = bundle.trajs.index(bundle.active)

    def _check_spawning(self, traj: Trajectory):
        # TODO: add different options
        mol = traj.mol
        for s in range(mol.n_states):
            if s in traj.mol.disabled:
                continue
            if np.abs(mol.nacdt_ss[mol.state, s]) > self._csthresh:
                self._spawnstate = s
                break

    def _enter_spawning(self, traj: Trajectory):
        # TODO: adjust timestep
        # TODO: save steps
        if self._spawnprop:
            return
        traj.mol.disabled.extend([traj.mol.state, self._spawnstate])
        self.spawn_traj(traj, traj.mol.state)
        self._spawnprop = self._clone
        self._tentry = traj.timestep.time

    def _exit_spawning(self):
        self._spawnprop = None
        self._spawnstate = None

    def _check_backprop(self, traj: Trajectory):
        return traj.timestep.time <= self._tentry

    def _enter_backprop(self, traj: Trajectory):
        traj.mols[-1] = traj.mols[-3]
        self._backprop = traj
        self._backprop.timestep.dt *= -1
        self._backprop.mol.state = self._spawnstate

    def _exit_backprop(self):
        self._backprop.mols[-1] = self._backprop.mols[-3]
        self._backprop.timestep.dt *= -1
        self._backprop = None

    def spawn_traj(self, traj: Trajectory, state: int):
        clone = deepcopy(traj)
        clone.mol.state = state
        clone.mol.amp = 0j
        self._clone = clone

    def update_traj(self, traj: Trajectory):
        ts = traj.timestep
        mol = traj.mols[-1]
        mol.phase += 0.5 * mol.kinetic_energy() * ts.dt
        super().update_traj(traj)

        mol = traj.mols[-1]
        mol.phase += 0.5 * mol.kinetic_energy() * ts.dt

    def adjust_nuclear(self, traj: Trajectory):
        coup = lambda mol: np.abs(mol.nacdt_ss[mol.state, self._spawnstate])

        mols = traj.mols

        if self._spawnprop and (coup(mols[-3]) < coup(mols[-2]) and coup(mols[-1]) < coup(mols[-2])):
            fm.log(f"Spawning on state {self._spawnstate}.")
            self._enter_backprop(self._spawnprop)
            self._exit_spawning()
            self._tspawn = traj.timestep.time

    def update_coeff(self, mols, ts):
        pass

    def update_amps(self, mols_ini: list[Molecule], mols_fin: list[Molecule], ts: Timestep):
        cg_ini = CG(*mols_ini)
        cg_fin = CG(*mols_fin)

        heff_ini = cg_ini.tbf_heff()
        heff_fin = cg_fin.tbf_heff()

        amp = [mol.amp for mol in mols_ini]

        nsub = 50
        for i in range(nsub):
            frac = (i + 0.5) / nsub
            heff = (1 - frac) * heff_ini + frac * heff_fin
            amp = expm(-1j*heff * ts.dt / nsub) @ amp

        print("NORM: ", np.einsum("i,ij,j->", amp.conj(), cg_fin.tbf_ovl(), amp))

        for i, mol in enumerate(mols_fin):
            mol.amp = amp[i]

@Config.Option("aims", "mixin")
class AIMSMixin(MoleculeMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = self._state
        self.coeff_s[self.state] = 1
        self.phase = 0
        self.amp = 1 + 0j
        self.disabled = []

        self.alpha = np.array([atomic_widths[x] for x in self.name_a.astype("<U2")])

    @property
    def active(self):
        return self.state

    def dat_header(self):
        dic = {"phs": fmt("Phase")}
        return dic | super().dat_header()

    def dat_dic(self):
        dic = {"phs": fmt(self.phase)}
        return dic | super().dat_dic()

    def h5_dict(self):
        dic = {"phase": self.phase}
        return dic | super().h5_dict()

class CG:
    def __init__(self, *mols: Molecule):
        nst, nat, ndim = mols[0].grad_sad.shape
        ndof = nat*ndim
        self.mass = np.tile(mols[0].mass_a[:,None], ndim).flatten()
        self.wid = np.tile(mols[0].alpha[:,None], ndim).flatten()

        self.pos = np.array([mol.pos_ad.flatten() for mol in mols])
        self.vel = np.array([mol.vel_ad.flatten() for mol in mols])
        self.acc = np.array([mol.acc_ad.flatten() for mol in mols])

        self.coeff = np.array([mol.coeff_s for mol in mols])
        self.ham = np.array([mol.ham_eig_ss for mol in mols])
        self.grad = np.array([mol.grad_sad.reshape((nst, ndof)) for mol in mols])
        self.tdc = np.array([mol.nacdt_ss for mol in mols])
        np.nan_to_num(self.tdc, copy=False)
        self.phs = np.array([mol.phase for mol in mols])

        self.dpos =  self.pos[None,:] - self.pos[:,None]
        self.mpos = (self.pos[None,:] + self.pos[:,None])/2
        self.dmom =  self.mom[None,:] - self.mom[:,None]
        self.mmom = (self.mom[None,:] + self.mom[:,None])/2
        self.dphs =  self.phs[None,:] - self.phs[:,None]

        self.ovl = self.nuc_ovl()

    @property
    def mom(self):
        return self.vel * self.mass

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
        return np.einsum("mi,ni->mn", self.coeff.conj(), self.coeff) * self.ovl

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
        return np.einsum("mi,nj,mnij->mn", self.coeff.conj(), self.coeff, ham)

    def tbf_ddt(self):
        res = np.einsum("mi,ni->mn", self.coeff.conj(), self.coeff) * self.nuc_ddt()
        amp_dot = -np.einsum("mij,mj->mi", 1j * self.ham + self.tdc, self.coeff)
        res += np.einsum("mi,ni->mn", self.coeff.conj(), amp_dot) * self.ovl
        return res

    def tbf_heff(self):
        ovl_inv = np.linalg.inv(self.tbf_ovl())
        return ovl_inv @ (self.tbf_ham() - 1j*self.tbf_ddt())