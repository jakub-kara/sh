import numpy as np
from scipy.linalg import expm
from abc import ABC, abstractmethod
from classes.molecule import Molecule

class TDCUpdater(ABC):
    mode = None

    def is_ready(self, step: int):
        return True

    @abstractmethod
    def update(self, mol: list[Molecule], *args, **kwargs):
        raise NotImplementedError

    def no_update(self, mol: list[Molecule]):
        mol[-1].pes.nacdt_ss = mol[-2].pes.nacdt_ss

    def interpolate(self, mol: list[Molecule], frac: float):
        return frac * mol[-1].pes.nacdt_ss + (1 - frac) * mol[-2].pes.nacdt_ss

    def fix_phase(self, mol: Molecule):
        for i in range(mol.n_states):
            mol.pes.ovlp_ss[i,:] *= mol.pes.phase_s[i]

        phase_vec = np.ones(mol.n_states)
        for i in range(mol.n_states):
            if mol.pes.ovlp_ss[i,i] < 0:
                phase_vec[i] *= -1
                mol.pes.ovlp_ss[:,i] *= -1
        mol.pes.phase_s = phase_vec

class HST(TDCUpdater):
    # Classic Hammes-Schiffer-Tully mid point approximation (paper in 1994)
    def __init__(self, *args, **kwargs):
        self._nqsteps = 1
        self.mode = "o"

    def update(self, mol: list[Molecule], dt: float, *args, **kwargs):
        mol[-1].pes.nacdt_ss = 1 / (2 * dt) * (mol[-1].pes.ovlp_ss - mol[-1].pes.ovlp_ss.T)

    def interpolate(self, mol: list[Molecule], frac: float):
        return mol[-1].pes.nacdt_ss

class HSTSharc(TDCUpdater):
    # SHARC HST end-point finite difference, linearly interpolated across the region
    # Maybe don't trust this code too much...
    def __init__(self, *args, **kwargs):
        self._nqsteps = 1
        self.mode = "o"

    def is_ready(step: int):
        return step >= 2

    def update(self, mol: list[Molecule], dt: float, *args, **kwargs):
        mol[-1].pes.nacdt_ss = 1 / (4 * dt) * (3 * (mol[-1].pes.ovlp_ss - mol[-1].pes.ovlp_ss.T) - (mol[-2].pes.ovlp_ss - mol[-2].pes.ovlp_ss.T))

class NACME(TDCUpdater):
    # ddt = nacme . velocity (i.e. original Tully 1990 paper model)
    def __init__(self, *args, **kwargs):
        self._nqsteps = 1
        self.mode = "n"

    def update(self, mol: list[Molecule], *args, **kwargs):
        mol[-1].pes.nacdt_ss = np.einsum("ijad, ad -> ij", mol[-1].pes.nacdr_ssad, mol[-1].vel_ad)

class NPI(TDCUpdater):
    # Meek and Levine's norm preserving interpolation, but integrated across the time-step
    def __init__(self, *args, **kwargs):
        self._nqsteps = 1
        self.mode = "o"

    def update(self, mol: list[Molecule], dt: int, *args, **kwargs):
        nst = mol[-1].n_states
        U =    np.eye(nst)      *   np.cos(np.arccos(mol[-1].pes.ovlp_ss))
        U -=  (np.eye(nst) - 1) *   np.sin(np.arcsin(mol[-1].pes.ovlp_ss))
        dU =   np.eye(nst)      * (-np.sin(np.arccos(mol[-1].pes.ovlp_ss)) * np.arccos(mol[-1].pes.ovlp_ss) / dt)
        dU -= (np.eye(nst) - 1) *  (np.cos(np.arcsin(mol[-1].pes.ovlp_ss)) * np.arcsin(mol[-1].pes.ovlp_ss) / dt)

        mol[-1].pes.nacdt_ss = (U.T @ dU) * -1*(np.eye(nst) - 1) # to get rid of non-zero diagonal elements

class NPISharc(TDCUpdater):
    # NPI sharc mid-point averaged
    def __init__(self, n_qsteps: int):
        self._nqsteps = n_qsteps
        self.mode = "o"

    def update(self, mol: list[Molecule], dt: float, *args, **kwargs):
        nst = mol[-1].n_states
        Utot = np.zeros((nst, nst))
        for i in range(self._nqsteps):
            U   =  np.eye(nst)      *   np.cos(np.arccos(mol[-1].pes.ovlp_ss) * i / self._nqsteps)
            U  -= (np.eye(nst) - 1) *   np.sin(np.arcsin(mol[-1].pes.ovlp_ss) * i / self._nqsteps)
            dU  =  np.eye(nst)      * (-np.sin(np.arccos(mol[-1].pes.ovlp_ss) * i / self._nqsteps) * np.arccos(mol[-1].pes.ovlp_ss) / dt)
            dU -= (np.eye(nst) - 1) *  (np.cos(np.arcsin(mol[-1].pes.ovlp_ss) * i / self._nqsteps) * np.arcsin(mol[-1].pes.ovlp_ss) / dt)
            Utot += np.matmul(U.T, dU)

        Utot /= self._nqsteps
        mol[-1].pes.nacdt_ss = -Utot * (np.eye(nst) - 1)

class NPIMeek(TDCUpdater):
    # NPI Meek and Levine mid-point averaged
    def __init__(self, *args, **kwargs):
        self._nqsteps = 1
        self.mode = "o"

    def is_ready(self, step: int):
        return step > 2

    def update(self, mol: list[Molecule], dt: float, **kwargs):
        def sinc(x):
            if np.abs(x) < 1e-9:
                return 1
            else:
                return np.sin(x) / x

        nst = mol[-1].n_states
        w = mol[-1].pes.ovlp_ss
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

                mol[-1].pes.nacdt_ss[k,j] = 1 / (2 * dt) * (np.arccos(w[j,j]) * (A + B) + np.arcsin(w[k,j]) * (C + D) + E)

class LD(TDCUpdater):
    def __init__(self, n_qsteps: int, *args, **kwargs):
        self._nqsteps = n_qsteps
        self.mode = "o"
        self.rmat = None

    def is_ready(self, step: int):
        return step >= 2

    def update(self, mol: list[Molecule], dt: float, *args, **kwargs):
        nst = mol[-1].n_states
        R = np.eye(nst)
        H_tr = mol[-1].pes.ovlp_ss @ mol[-1].pes.ham_eig_ss @ mol[-1].pes.ovlp_ss.T
        for i in range(self._nqsteps):
            frac = (i + 1) / self._nqsteps
            H = frac * (H_tr - mol[-2].pes.ham_eig_ss) + mol[-2].pes.ham_eig_ss
            R = expm(-1j*H * dt / self._nqsteps) @ R
        self.rmat = mol[-1].pes.ovlp_ss.T @ R

    def no_update(self, mol):
        super().no_update(mol)
        self.rmat = np.eye(mol[-1].n_states)