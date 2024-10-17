import numpy as np
from scipy.linalg import expm
from dynamics.sh.sh import SurfaceHopping, HoppingChecker
from classes.molecule import Molecule
from integrators import select_tdcupd
from integrators.tdc import TDCUpdater
from integrators.qtemp import QuantumUpdater

class unSMASH(SurfaceHopping):
    def __init__(self, config: dict):
        super().__init__(config)
        self._decoherence = SurfaceHopping._decoherence_none

        dyn = config["dynamics"]
        active = dyn["initstate"]
        self._hopchck: MASHChecker = MASHChecker(self.n_states, active)

        qupd = HopUpdater(config["quantum"], self._hopchck)
        self.bind_quantum_updater(qupd)

    def adjust_nuclear(self):
        if self._hopchck.hop_ready():
            if self._has_energy(self.mol):
                self._hopchck.swap_bloch()
                self._adjust_velocity(self.mol)
                self._hop()
                self.run_est(self.mol, "a")
                self._nucupd.to_init()
            else:
                print("FRUSTRATED")
                self._hopchck.reverse_bloch()
                self._reverse_velocity(self.mol)
                self._nohop()

    def _has_energy(self, mol: Molecule):
        d2 = np.sum(mol.pes.nacdr_ssad[self.target, self.active]**2)
        pmw = mol.vel_ad * np.sqrt(mol.mass_a[:,None])
        ppar = np.sum(pmw * mol.pes.nacdr_ssad[self.target, self.active]) / d2 * mol.pes.nacdr_ssad[self.target, self.active]
        return np.sum(ppar**2) / 2 + mol.pes.ham_eig_ss[self.active, self.active] - mol.pes.ham_eig_ss[self.target, self.target] >= 0

    def _adjust_velocity(self, mol: Molecule):
        d2 = np.sum(mol.pes.nacdr_ssad[self.target, self.active]**2)
        pmw = mol.vel_ad * np.sqrt(mol.mass_a[:,None])
        ppar = np.sum(pmw * mol.pes.nacdr_ssad[self.target, self.active]) / d2 * mol.pes.nacdr_ssad[self.target, self.active]
        pperp = pmw - ppar
        pfin = np.sqrt(1 + 2 * (mol.pes.ham_eig_ss[self.active, self.active] - mol.pes.ham_eig_ss[self.target, self.target]) / np.sum(ppar**2)) * ppar
        mol.vel_ad = (pperp + pfin) / np.sqrt(mol.mass_a[:,None])

    def _reverse_velocity(self, mol: Molecule):
        d2 = np.sum(mol.pes.nacdr_ssad[self.target, self.active]**2)
        pmw = mol.vel_ad * np.sqrt(mol.mass_a[:,None])
        ppar = np.sum(pmw * mol.pes.nacdr_ssad[self.target, self.active]) / d2 * mol.pes.nacdr_ssad[self.target, self.active]
        pperp = pmw - ppar
        pfin = -ppar
        mol.vel_ad = (pperp + pfin) / np.sqrt(mol.mass_a[:,None])

    def propagate(self):
        super().propagate()
        print(self._hopchck._bloch)
        print(self.mol.pes.nacdt_ss[0,1])

class MASHChecker(HoppingChecker):
    def __init__(self, n_states, active):
        super().__init__(n_states, active)
        self._bloch = np.zeros((n_states, 3))
        self._bloch[:,2] = 1
        self._bloch[self.active] = None

    def get_target(self, *args, **kwargs):
        for s in range(self.n_states):
            if s == self.active:
                continue
            if self._bloch[s,2] < 0:
                self._target = s
                break

    def update(self, mol: list[Molecule], tdcupd: TDCUpdater, dtq: float, frac: float, *args, **kwargs):
        act = self.active
        ham = frac * mol[-1].pes.ham_eig_ss + (1 - frac) * mol[-2].pes.ham_eig_ss
        for s in range(self.n_states):
            if s == act:
                continue

            mat = np.zeros((3, 3))
            mat[0,1] = ham[s, s] - ham[act, act]
            mat[1,0] = -mat[0,1]
            mat[0,2] = 2 * tdcupd.interpolate(mol, frac)[act, s]
            mat[2,0] = -mat[0,2]

            self._bloch[s] = expm(mat * dtq) @ self._bloch[s]

    def swap_bloch(self):
        swp = np.array([1, -1, -1])
        for s in range(self.n_states):
            if s == self.active:
                self._bloch[s] = self._bloch[self._target] * swp
                self._bloch[self._target] = None

    def reverse_bloch(self):
        self._bloch[self._target, 2] *= -1

class HopUpdater(QuantumUpdater):
    def __init__(self, config, hop: MASHChecker):
        self._nqsteps = config["nqsteps"]
        self._tdcupd: TDCUpdater = select_tdcupd("nacme")(self._nqsteps)
        self._hop = hop

    def update(self, mol: list[Molecule], dt: float, step: int, *args, **kwargs):
        self._tdcupd.fix_phase(mol[-1])

        if self._tdcupd.is_ready(step):
            self._tdcupd.update(mol, dt)
        else:
            self._tdcupd.no_update(mol)

        for i in range(self._nqsteps):
            frac = (i + 0.5) / self._nqsteps
            self._hop.update(mol, self._tdcupd, dt / self._nqsteps, frac)
        self._hop.get_target()