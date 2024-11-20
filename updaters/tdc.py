import numpy as np
from .updaters import Updater, Multistage, UpdateResult
from classes.meta import SingletonFactory
from classes.molecule import Molecule

class TDCUpdater(Updater, metaclass = SingletonFactory):
    mode = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tdc = None

    def new_result(self, mol):
        self.tdc = UpdateResult(mol.nacdt_ss, self.substeps)

    def no_update(self, mols: list[Molecule], dt: float):
        self.tdc.fill()

class kTDCe(TDCUpdater, key = "ktdce"):
    # curvature-based TDC approximation, energy version
    mode = ""
    steps = 4

    def _dh(self, mol: Molecule, i: int, j: int):
        return mol.ham_eig_ss[i,i] - mol.ham_eig_ss[j,j]

    def update(self, mols: list[Molecule], dt: float):
        tdc = self.tdc.inp
        for i in range(mols[-1].n_states):
            for j in range(i):
                dh = lambda mol: self._dh(mol, i, j)
                d2dh = 1/dt**2 * (2 * dh(mols[-1]) - 5 * dh(mols[-2]) + 4 * dh(mols[-3]) - dh(mols[-4]))
                rad = d2dh / dh(mols[-1])
                if rad > 0:
                    tdc[i,j] = 1/2 * np.sqrt(rad)
                tdc[j,i] = -tdc[i,j]
        self.tdc.out = tdc

class kTDCg(TDCUpdater, key =  "ktdcg"):
    # curvature-based TDC approximation, gradient version
    mode = "g"
    steps = 2

    def _dh(self, mol: Molecule, i: int, j: int):
        return mol.ham_eig_ss[i,i] - mol.ham_eig_ss[j,j]

    def _ddh(self, mol: Molecule, i: int, j: int):
        return np.sum((mol.grad_sad[i] - mol.grad_sad[j]) * mol.vel_ad)

    def update(self, mols: list[Molecule], dt: float):
        tdc = self.tdc.inp
        for i in range(mols[-1].n_states):
            for j in range(i):
                ddh = lambda mol: self._ddh(mol, i, j)
                d2dh = 1/dt * (ddh(mols[-1]) - ddh(mols[-2]))
                rad = d2dh / self._dh(mols[-1], i, j)
                if rad > 0:
                    tdc[i,j] = 1/2 * np.sqrt(rad)
                tdc[j,i] = -tdc[i,j]
        self.tdc.out = tdc

class HST(TDCUpdater, key = "hst"):
    # Classic Hammes-Schiffer-Tully mid point approximation (paper in 1994)
    mode = "o"
    steps = 2

    def update(self, mols: list[Molecule], dt: float):
        self.tdc.out = 1 / (2 * dt) * (mols[-1].ovlp_ss - mols[-1].ovlp_ss.T)

class HSTSharc(TDCUpdater, key = "hstsharc"):
    # SHARC HST end-point finite difference, linearly interpolated across the region
    # Maybe don't trust this code too much...
    mode = "o"
    steps = 3

    def update(self, mols: list[Molecule], dt: float):
        self.tdc.out = 1 / (4 * dt) * (3 * (mols[-1].ovlp_ss - mols[-1].ovlp_ss.T) - (mols[-2].ovlp_ss - mols[-2].ovlp_ss.T))

class NACME(TDCUpdater, key = "nacme"):
    # ddt = nacme . velocity (i.e. original Tully 1990 paper model)
    mode = "n"
    steps = 1

    def update(self, mols: list[Molecule], dt: float):
        self.tdc.out = np.einsum("ijad, ad -> ij", mols[-1].nacdr_ssad, mols[-1].vel_ad)

class NPI(TDCUpdater, key = "npi"):
    # Meek and Levine's norm preserving interpolation, but integrated across the time-step
    mode = "o"
    steps = 2

    def update(self, mols: list[Molecule], dt: float):
        nst = mols[-1].n_states
        U =    np.eye(nst)      *   np.cos(np.arccos(mols[-1].ovlp_ss))
        U -=  (np.eye(nst) - 1) *   np.sin(np.arcsin(mols[-1].ovlp_ss))
        dU =   np.eye(nst)      * (-np.sin(np.arccos(mols[-1].ovlp_ss)) * np.arccos(mols[-1].ovlp_ss) / dt)
        dU -= (np.eye(nst) - 1) *  (np.cos(np.arcsin(mols[-1].ovlp_ss)) * np.arcsin(mols[-1].ovlp_ss) / dt)

        self.tdc.out = (U.T @ dU) * (1 - np.eye(nst)) # to get rid of non-zero diagonal elements

class NPISharc(Multistage, TDCUpdater, key = "npisharc"):
    # NPI sharc mid-point averaged
    mode = "o"
    steps = 2

    def update(self, mols: list[Molecule], dt: float):
        nst = mols[-1].n_states
        Utot = np.zeros((nst, nst))
        for i in range(self.substeps):
            U   =  np.eye(nst)      *   np.cos(np.arccos(mols[-1].ovlp_ss) * i / self.substeps)
            U  -= (np.eye(nst) - 1) *   np.sin(np.arcsin(mols[-1].ovlp_ss) * i / self.substeps)
            dU  =  np.eye(nst)      * (-np.sin(np.arccos(mols[-1].ovlp_ss) * i / self.substeps) * np.arccos(mols[-1].ovlp_ss) / dt)
            dU -= (np.eye(nst) - 1) *  (np.cos(np.arcsin(mols[-1].ovlp_ss) * i / self.substeps) * np.arcsin(mols[-1].ovlp_ss) / dt)

            Utot += np.matmul(U.T, dU)
            self.tdc.inter[i] = Utot / (i + 1) * (1 - np.eye(nst))

class NPIMeek(TDCUpdater, key = "npimeek"):
    # NPI Meek and Levine mid-point averaged
    mode = "o"
    steps = 2

    def update(self, mols: list[Molecule], dt: float, **kwargs):
        def sinc(x):
            if np.abs(x) < 1e-9:
                return 1
            else:
                return np.sin(x) / x

        tdc = self.tdc.inp
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

                tdc[k,j] = 1 / (2 * dt) * (np.arccos(w[j,j]) * (A + B) + np.arcsin(w[k,j]) * (C + D) + E)
        self.tdc.out = tdc