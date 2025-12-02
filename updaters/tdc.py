import numpy as np
from .base import Updater, Multistage, UpdateResult
from classes.config import Config, get_cfg
from classes.meta import Singleton
from classes.molecule import Molecule
from classes.timestep import Timestep
from electronic.base import ESTMode

@Config("tdc_upd", default="npi", path=Config.QUANTUM, valid=Config.STR)
class TDCUpdater(Updater, metaclass = Singleton):
    mode = ESTMode(ESTMode.NONE)

    def new_result(self, mol):
        self.result = UpdateResult(mol.nacdt_ss, self.substeps)

    def _validate_overlap(self, ovl):
        eps = 1e-12
        ovl[ovl > 1 - eps] = 1 - eps

@Config.Option("none", "tdc_upd")
class BlankTDCUpdater(TDCUpdater):
    def update(self, mols, dt):
        self.no_update(mols, dt)

@Config.Option("ktdce", "tdc_upd")
class kTDCe(TDCUpdater):
    # curvature-based TDC approximation, energy version
    steps = 4

    def _dh(self, mol: Molecule, i: int, j: int):
        return mol.ham_eig_ss[i,i] - mol.ham_eig_ss[j,j]

    def update(self, mols: list[Molecule], ts: Timestep):
        tdc = self.result.inp
        for i in range(mols[-1].n_states):
            for j in range(i):
                dh = lambda mol: self._dh(mol, i, j)
                d2dh = 1/ts.dt**2 * (2 * dh(mols[-1]) - 5 * dh(mols[-2]) + 4 * dh(mols[-3]) - dh(mols[-4]))
                rad = d2dh / dh(mols[-1])
                if rad > 0:
                    tdc[i,j] = 1/2 * np.sqrt(rad)
                tdc[j,i] = -tdc[i,j]
        self.result.push(tdc)

@Config.Option("ktdcg", "tdc_upd")
class kTDCg(TDCUpdater):
    # curvature-based TDC approximation, gradient version
    mode = ESTMode(ESTMode.GRADIENTS)
    steps = 2

    def _dh(self, mol: Molecule, i: int, j: int):
        return mol.ham_eig_ss[i,i] - mol.ham_eig_ss[j,j]

    def _ddh(self, mol: Molecule, i: int, j: int):
        return np.sum((mol.grad_sad[i] - mol.grad_sad[j]) * mol.vel_ad)

    def update(self, mols: list[Molecule], ts: Timestep):
        tdc = self.result.inp
        for i in range(mols[-1].n_states):
            for j in range(i):
                ddh = lambda mol: self._ddh(mol, i, j)
                d2dh = 1/ts.dt * (ddh(mols[-1]) - ddh(mols[-2]))
                rad = d2dh / self._dh(mols[-1], i, j)
                if rad > 0:
                    tdc[i,j] = 1/2 * np.sqrt(rad)
                tdc[j,i] = -tdc[i,j]
        self.result.push(tdc)

@Config.Option("hst", "tdc_upd")
class HST(TDCUpdater):
    # Classic Hammes-Schiffer-Tully mid point approximation (paper in 1994)
    mode = ESTMode(ESTMode.OVERLAP)
    steps = 2

    def update(self, mols: list[Molecule], ts: Timestep):
        self._validate_overlap(mols[-1].ovlp_ss)
        out = 1 / (2 * ts.dt) * (mols[-1].ovlp_ss - mols[-1].ovlp_ss.T)
        self.result.push(out)

@Config.Option("hst3", "tdc_upd")
class HSTSharc(TDCUpdater):
    mode = ESTMode(ESTMode.OVERLAP)
    steps = 3

    def update(self, mols: list[Molecule], ts: Timestep):
        self._validate_overlap(mols[-1].ovlp_ss)
        out = 1 / (4 * ts.dt) * (3 * (mols[-1].ovlp_ss - mols[-1].ovlp_ss.T) - (mols[-2].ovlp_ss - mols[-2].ovlp_ss.T))
        self.result.push(out)

@Config.Option("nacme", "tdc_upd")
class NACME(TDCUpdater):
    # ddt = nacme . velocity (i.e. original Tully 1990 paper model)
    mode = ESTMode(ESTMode.NACMES)
    steps = 1

    def update(self, mols: list[Molecule], ts: Timestep):
        temp = np.nan_to_num(mols[-1].nacdr_ssad)
        self.result.push(np.einsum("ijad, ad -> ij", temp, mols[-1].vel_ad))

@Config.Option("npi", "tdc_upd")
class NPI(Multistage, TDCUpdater):
    # Meek and Levine's norm preserving interpolation, but integrated across the time-step
    mode = ESTMode(ESTMode.OVERLAP)
    steps = 1

    def update(self, mols: list[Molecule], ts: Timestep):
        self._validate_overlap(mols[-1].ovlp_ss)
        nst = mols[-1].n_states
        for i in range(self.substeps):
            U   =  np.eye(nst)      *   np.cos(np.arccos(mols[-1].ovlp_ss) * i / self.substeps)
            U  -= (np.eye(nst) - 1) *   np.sin(np.arcsin(mols[-1].ovlp_ss) * i / self.substeps)
            dU  =  np.eye(nst)      * (-np.sin(np.arccos(mols[-1].ovlp_ss) * i / self.substeps) * np.arccos(mols[-1].ovlp_ss) / ts.dt)
            dU -= (np.eye(nst) - 1) *  (np.cos(np.arcsin(mols[-1].ovlp_ss) * i / self.substeps) * np.arcsin(mols[-1].ovlp_ss) / ts.dt)

            Utot = np.matmul(U.T, dU)
            self.result.push(Utot * (1 - np.eye(nst)))

@Config.Option("npisharc", "tdc_upd")
class NPISharc(Multistage, TDCUpdater):
    # NPI sharc mid-point averaged
    mode = ESTMode(ESTMode.OVERLAP)
    steps = 1

    def update(self, mols: list[Molecule], ts: Timestep):
        nst = mols[-1].n_states
        self._validate_overlap(mols[-1].ovlp_ss)
        Utot = np.zeros((nst, nst))
        for i in range(self.substeps):
            U   =  np.eye(nst)      *   np.cos(np.arccos(mols[-1].ovlp_ss) * i / self.substeps)
            U  -= (np.eye(nst) - 1) *   np.sin(np.arcsin(mols[-1].ovlp_ss) * i / self.substeps)
            dU  =  np.eye(nst)      * (-np.sin(np.arccos(mols[-1].ovlp_ss) * i / self.substeps) * np.arccos(mols[-1].ovlp_ss) / ts.dt)
            dU -= (np.eye(nst) - 1) *  (np.cos(np.arcsin(mols[-1].ovlp_ss) * i / self.substeps) * np.arcsin(mols[-1].ovlp_ss) / ts.dt)

            Utot += np.matmul(U.T, dU)
            self.result.push(Utot / (i + 1) * (1 - np.eye(nst)))

@Config.Option("npimeek", "tdc_upd")
class NPIMeek(TDCUpdater):
    # NPI Meek and Levine mid-point averaged
    mode = ESTMode(ESTMode.OVERLAP)
    steps = 2

    def update(self, mols: list[Molecule], ts: Timestep, **kwargs):
        def sinc(x):
            if np.abs(x) < 1e-9:
                return 1
            else:
                return np.sin(x) / x

        self._validate_overlap(mols[-1].ovlp_ss)
        tdc = self.result.inp
        nst = mols[-1].n_states
        w = mols[-1].ovlp_ss

        for k in range(nst):
            for j in range(nst):
                if k == j:
                    continue
                A = -sinc(np.arccos(w[j,j]) - np.arcsin(w[j,k]))
                B =  sinc(np.arccos(w[j,j]) + np.arcsin(w[j,k]))
                C =  sinc(np.arccos(w[k,k]) - np.arcsin(w[k,j]))
                D =  sinc(np.arccos(w[k,k]) + np.arcsin(w[k,j]))
                E = 0.
                if nst != 2:
                    sqarg = 1 - w[j,j]**2 - w[k,j]**2
                    if sqarg > 1e-6:
                        wlj = np.sqrt(sqarg)
                        wlk = -(w[j,k] * w[j,j] + w[k,k] * w[k,j]) / wlj
                        if np.abs(wlk - wlj) > 1e-6:
                            E = wlj**2
                        else:
                            E = 2 * np.arcsin(wlj) * (wlj*wlk*np.arcsin(wlj) + (np.sqrt((1 - wlj**2)*(1 - wlk**2)) - 1) * np.arcsin(wlk))
                            E /= (np.arcsin(wlj)**2 - np.arcsin(wlk)**2)

                tdc[k,j] = 1 / (2 * ts.dt) * (np.arccos(w[j,j]) * (A + B) + np.arcsin(w[k,j]) * (C + D) + E)
        self.result.push(tdc)
