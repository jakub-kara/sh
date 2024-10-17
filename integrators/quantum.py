import numpy as np
from abc import abstractmethod, ABC
from scipy.linalg import expm
from classes.molecule import Molecule

class Interpolants:
    def __init__(self, n_qsteps, n_states):
        self._tdc = np.zeros((n_qsteps, n_states, n_states))
        self._coeff = np.zeros((n_qsteps, n_states), dtype=np.complex128)

    @property
    def n_qsteps(self):
        return self._coeff.shape[0]

    @property
    def n_states(self):
        return self._coeff.shape[1]

    @property
    def tdc(self):
        return self._tdc

    @tdc.setter
    def tdc(self, val):
        self._tdc[:] = val

    @property
    def coeff(self):
        return self._coeff

    @coeff.setter
    def coeff(self, val):
        self._coeff[:] = val


# TODO: move rmat functionality to tdcupd
class LDInterpolants(Interpolants):
    def __init__(self, n_qsteps, n_states):
        super().__init__(n_qsteps, n_states)
        self._rmat = np.zeros((n_states, n_states), dtype=np.complex128)

    @property
    def rmat(self):
        return self._rmat

    @rmat.setter
    def rmat(self, val):
        self._rmat[:] = val

class TDCUpdater(ABC):
    tdc = None
    mode = None

    def is_ready(self, step: int):
        return True

    @abstractmethod
    def update(self, mol: list[Molecule], inter: Interpolants, dt):
        raise NotImplementedError

    def no_update(self, mol: list[Molecule], inter: Interpolants):
        inter.tdc[:] = mol[-2].pes.nacdt_ss

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
    mode = "o"

    def update(self, mol: list[Molecule], inter: Interpolants, dt):
        self.tdc = 1 / (2 * dt) * (mol[-1].pes.ovlp_ss - mol[-1].pes.ovlp_ss.T)
        inter.tdc[:] = self.tdc

class HSTSharc(TDCUpdater):
    # SHARC HST end-point finite difference, linearly interpolated across the region
    # Maybe don't trust this code too much...
    mode = "o"

    def is_ready(step: int):
        return step >= 2

    def update(self, mol: list[Molecule], inter: Interpolants, dt):
        ddt_ini = 1 / (4 * dt) * (3 * (mol[-2].pes.ovlp_ss - mol[-2].pes.ovlp_ss.T) - (mol[-3].pes.ovlp_ss - mol[-3].pes.ovlp_ss.T))
        ddt_fin = 1 / (4 * dt) * (3 * (mol[-1].pes.ovlp_ss - mol[-1].pes.ovlp_ss.T) - (mol[-2].pes.ovlp_ss - mol[-2].pes.ovlp_ss.T))
        for i in range(inter.n_qsteps):
            frac = (i + 0.5) / inter.n_qsteps
            inter.tdc[i] = frac * ddt_ini + (1 - frac) * ddt_fin
        self.tdc = inter.tdc[-1]

class NACME(TDCUpdater):
    # ddt = nacme . velocity (i.e. original Tully 1990 paper model)
    mode = "n"

    def update(self, mol: list[Molecule], inter: Interpolants, dt):
        self.tdc = np.einsum("ijad, ad -> ij", mol[-1].pes.nacdr_ssad, mol[-1].vel_ad)
        nst = mol[-1].n_states
        temp = np.zeros((nst, nst))
        for s1 in range(nst):
            for s2 in range(s1):
                temp[s1,s2] = np.sum(mol[-1].vel_ad * mol[-1].pes.nacdr_ssad[s1,s2])
                temp[s2,s1] = -temp[s1,s2]

        for i in range(inter.n_qsteps):
            frac = (i + 0.5) / inter.n_qsteps
            inter.tdc[i] = frac * self.tdc + (1 - frac) * mol[-2].pes.nacdt_ss

class NPI(TDCUpdater):
    # Meek and Levine's norm preserving interpolation, but integrated across the time-step
    mode = "o"

    def update(self, mol: list[Molecule], inter: Interpolants, dt):
        for i in range(inter.n_qsteps):
            frac = (i + 0.5) / inter.n_qsteps
            U =    np.eye(inter.n_states)      *   np.cos(np.arccos(mol[-1].pes.ovlp_ss) * frac)
            U -=  (np.eye(inter.n_states) - 1) *   np.sin(np.arcsin(mol[-1].pes.ovlp_ss) * frac)
            dU =   np.eye(inter.n_states)      * (-np.sin(np.arccos(mol[-1].pes.ovlp_ss) * frac) * np.arccos(mol[-1].pes.ovlp_ss) / dt)
            dU -= (np.eye(inter.n_states) - 1) *  (np.cos(np.arcsin(mol[-1].pes.ovlp_ss) * frac) * np.arcsin(mol[-1].pes.ovlp_ss) / dt)

            inter.tdc[i] = (U.T @ dU) * -1*(np.eye(inter.n_states) - 1) # to get rid of non-zero diagonal elements
        self.tdc = inter.tdc[-1]

class NPISharc(TDCUpdater):
    # NPI sharc mid-point averaged
    def update(self, mol: list[Molecule], inter: Interpolants, dt):
        Utot = np.zeros((inter.n_states, inter.n_states))
        for i in range(inter.n_qsteps):
            U   =  np.eye(inter.n_states)      *   np.cos(np.arccos(mol[-1].pes.ovlp_ss) * i / inter.n_qsteps)
            U  -= (np.eye(inter.n_states) - 1) *   np.sin(np.arcsin(mol[-1].pes.ovlp_ss) * i / inter.n_qsteps)
            dU  =  np.eye(inter.n_states)      * (-np.sin(np.arccos(mol[-1].pes.ovlp_ss) * i / inter.n_qsteps) * np.arccos(mol[-1].pes.ovlp_ss) / dt)
            dU -= (np.eye(inter.n_states) - 1) *  (np.cos(np.arcsin(mol[-1].pes.ovlp_ss) * i / inter.n_qsteps) * np.arcsin(mol[-1].pes.ovlp_ss) / dt)
            Utot += np.matmul(U.T, dU)

        Utot /= inter.n_qsteps
        self.tdc = -Utot * (np.eye(inter.n_states) - 1)
        inter.tdc[:] = self.tdc

class NPIMeek(TDCUpdater):
    # NPI Meek and Levine mid-point averaged
    def is_ready(self, step: int):
        return step > 2

    def update(self, mol: list[Molecule], inter: Interpolants, dt):
        def sinc(x):
            if np.abs(x) < 1e-9:
                return 1
            else:
                return np.sin(x) / x

        self.tdc = np.zeros((inter.n_states, inter.n_states))
        w = mol[-1].pes.ovlp_ss
        for k in range(inter.n_states):
            for j in range(inter.n_states):
                if k == j:
                    continue
                A = -sinc(np.arccos(w[j,j]) - np.arcsin(w[j,k]))
                B =  sinc(np.arccos(w[j,j]) + np.arcsin(w[j,k]))
                C =  sinc(np.arccos(w[k,k]) - np.arcsin(w[k,j]))
                D =  sinc(np.arccos(w[k,k]) + np.arcsin(w[k,j]))
                E = 0.
                if inter.n_states != 2:
                    sqarg = 1 - w[j,j]**2 - w[k,j]**2
                    if sqarg > 1e-6:
                        wlj = np.sqrt(sqarg)
                        wlk = -(w[j,k] * w[j,j] + w[k,k] * w[k,j]) / wlj
                        if np.abs(wlk - wlj) > 1e-6:
                            E = wlj**2
                        else:
                            E = 2 * np.arcsin(wlj) * (wlj*wlk*np.arcsin(wlj) + (np.sqrt((1 - wlj**2)*(1 - wlk**2)) - 1) * np.arcsin(wlk))
                            E /= (np.arcsin(wlj)**2 - np.arcsin(wlk)**2)

                self.tdc[k,j] = 1 / (2 * dt) * (np.arccos(w[j,j]) * (A + B) + np.arcsin(w[k,j]) * (C + D) + E)
        inter.tdc[:] = self.tdc

class LD(TDCUpdater):
    def is_ready(self, step: int):
        return step >= 2

    def update(self, mol: list[Molecule], inter: LDInterpolants, dt):
        R = np.eye(inter.n_states)
        H_tr = mol[-1].pes.ovlp_ss @ mol[-1].pes.ham_eig_ss @ mol[-1].pes.ovlp_ss.T
        for i in range(inter.n_qsteps):
            frac = (i + 1) / inter.n_qsteps
            H = frac * (H_tr - mol[-2].pes.ham_eig_ss) + mol[-2].pes.ham_eig_ss
            R = expm(-1j*H * dt / inter.n_qsteps) @ R
        inter.rmat = mol[-1].pes.ovlp_ss.T @ R

class CoeffUpdater:
    def update(self, mol: list[Molecule], inter: Interpolants, dt: float):
        # transform coeff into correct representation
        inter.coeff[-1] = mol[-1].pes.trans_ss @ mol[-1].pes.coeff_s

        # propagation
        for i in range(inter.n_qsteps):
            # create variables for argument
            frac = (i + 0.5) / inter.n_qsteps
            ham_ss = frac * mol[-1].pes.ham_eig_ss + (1 - frac) * mol[-2].pes.ham_eig_ss

            # we use precomputed ddts
            arg = -(1.j*ham_ss + inter.tdc[i]) * dt / inter.n_qsteps

            #propagate using matrix exponential propagator
            inter.coeff[i] = expm(arg) @ inter.coeff[i-1]

        # transform coeff back
        inter.coeff[-1] = mol[-1].pes.trans_ss.conj().T @ inter.coeff[-1]
        mol[-1].pes.coeff_s = inter.coeff[-1]

class LDCoeffUpdater(CoeffUpdater):
    def update(self, mol: list[Molecule], inter: LDInterpolants, dt: float):
        mol[-1].pes.coeff_s = inter.rmat @ mol[-2].pes.coeff_s
        inter.coeff[:] = mol[-1].pes.coeff_s[:]

class QuantumUpdater:
    def __init__(self, n_states: int, config):
        n_qsteps = config["nqsteps"]
        self._tdcupd: TDCUpdater = self._select_tdcupd(config["tdcupd"])()
        self._cupd: CoeffUpdater = self._select_cupd(config["cupd"])()
        if config["tdcupd"] == "ld":
            self._inter = LDInterpolants(n_qsteps, n_states)
        else:
            self._inter = Interpolants(n_qsteps, n_states)

        self._mode = "grad"
        if config["tdcupd"] in ["nacme"]:
            self._mode = "nacs"

    @property
    def n_qsteps(self):
        return self._inter.n_qsteps

    @property
    def mode(self):
        return self._mode

    @property
    def tdc_qss(self):
        return self._inter.tdc

    @property
    def coeff_qs(self):
        return self._inter.coeff

    @property
    def rmat(self):
        return self._inter.rmat

    def _select_cupd(self, key):
        items = {
            "propmat": CoeffUpdater,
            "rmat": LDCoeffUpdater,
        }
        return items[key]

    def _select_tdcupd(self, key):
        items = {
            "hst": HST,
            "hstsharc": HSTSharc,
            "nacme": NACME,
            "npi": NPI,
            "npimeek": NPIMeek,
            "npisharc": NPISharc,
            "ld": LD,
        }
        return items[key]

    def update(self, mol: list[Molecule], step: int, dt: float):
        self._tdcupd.fix_phase(mol[-1])
        if self._tdcupd.is_ready(step):
            self._tdcupd.update(mol, self._inter, dt)
            self._cupd.update(mol, self._inter, dt)
        else:
            self._tdcupd.no_update(mol, self._inter)
            mol[-1].pes.coeff_s = mol[-2].pes.coeff_s
        mol[-1].pes.nacdt_ss[:] = self._tdcupd.tdc
