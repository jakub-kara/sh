import enum
import numpy as np
from classes.molecule import Molecule, BlochMixin
from classes.meta import SingletonFactory
from updaters.updaters import Updater, Multistage, UpdateResult
from updaters.coeff import CoeffUpdater
from updaters.tdc import TDCUpdater

class HoppingUpdater(Updater, metaclass = SingletonFactory):
    def __init__(self, *, seed = None, **config):
        super().__init__(**config)
        if seed is None:
            self._seed = np.random.default_rng().integers(9223372036854775807)
        else:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        options = {
            "i" : self._check_hop_i,
            "c" : self._check_hop_c
        }

        self._check_hop = options[config.get("prob_type","i")]

        if self._check_hop == self._check_hop_c:
            self._r = None
            self._cum_prob = None


    def new_result(self, mol: Molecule, active: int):
        self.hop = UpdateResult(active, self.substeps)
        self.prob = UpdateResult(np.zeros(mol.n_states), self.substeps)

    def no_update(self, mols: list[Molecule], dt: float, active: int):
        self.hop.fill()

    def _check_hop_c(self, prob: np.ndarray, active : int, dt: float):
        if self._r is None:
            self._r= self._rng.random()
            self._cum_prob = 0.

        self._cum_prob += (1-self._cum_prob) * (1-np.exp(-dt*np.sum(prob)))

        if self._r < self._cum_prob:
            norm_prob = prob/np.sum(prob)
            a = self._rng.random()
            for s, p in enumerate(norm_prob):
                if a < p:
                    break
            self._r = None
            return s
        return active


    def _check_hop_i(self, prob: np.ndarray, active: int, dt: float):
        r = self._rng.random()
        cum_prob = 0
        for s, p in enumerate(prob):
            cum_prob += p
            if r < cum_prob:
                return s
        return active

class NoHoppingUpdater(HoppingUpdater):
    key = "none"
    steps = 1

    def update(self, mols, dt, *args, **kwargs):
        self.hop.fill()

class TDCHoppingChecker(Multistage, HoppingUpdater):
    ''' CLASSIC TULLY '''
    key = "tdc"
    steps = 1

    def update(self, mols: list[Molecule], dt: float, active: int):
        cupd = CoeffUpdater()
        tdcupd = TDCUpdater()
        nst = mols[-1].n_states

        prob = self.prob.inp
        target = self.hop.inp

        for i in range(self.substeps):
            frac = (i + 0.5) / self.substeps
            tdc = tdcupd.tdc.interpolate(frac)
            coeff = cupd.coeff.interpolate(frac)

            if target != active:
                self.hop.inter[i:] = self.hop.inter[i-1]
                self.prob.inter[i:] = self.prob.inter[i-1]
                return
            for s in range(nst):
                # assign 0 hopping probability to active state
                if s == active:
                    prob[s] = 0
                # standard Tully-based hopping probability
                else:
                    # TODO: check timestep changes with variable timestep
                    temp = np.real(tdc[s, active] * np.conj(coeff[active]) * coeff[s])
                    temp *= -2 * (dt/self.substeps) /  np.abs(coeff[active])**2
                    prob[s] = max(0, temp)
            self.prob.inter[i] = prob
            self.hop.inter[i] = self._check_hop(prob, active, dt/self.substeps)
            target = self.hop.inter[i]

class PropHoppingChecker(HoppingUpdater):
    key = "prop"
    steps = 2

    def update(self, mols: list[Molecule], dt: float, active: int):
        cupd = CoeffUpdater()
        nst = mols[-1].n_states

        prob = self.prob.inp

        for s in range(nst):
            if s == active:
                prob[s] = 0
            else:
                temp = (1 - np.abs(cupd.coeff.out[active])**2 / np.abs(cupd.coeff.inp[active])**2)
                temp *= np.real(cupd.coeff.out[s] * np.conj(cupd.prop.out[s, active]) * np.conj(cupd.coeff.inp[active]))
                temp /= (np.abs(cupd.coeff.inp[active])**2 - \
                         np.real(cupd.coeff.out[active] * np.conj(cupd.prop.out[active, active]) * np.conj(cupd.coeff.inp[active])))
                prob[s] = max(0, temp)
        self.prob.out = prob
        self.hop.out = self._check_hop(prob, active, dt)

class GFHoppingChecker(HoppingUpdater):
    key = "gf"
    steps = 2

    def update(self, mols: list[Molecule], dt: float, active: int):
        cupd = CoeffUpdater()
        nst = mols[-1].n_states
        prob = self.prob.inp

        fact = (1 - np.abs(cupd.coeff.out[active])**2 / np.abs(cupd.coeff.inp[active])**2)
        fact /= np.sum(np.maximum(0, np.abs(cupd.coeff.out)**2 - np.abs(cupd.coeff.inp)**2))
        for s in range(nst):
            if s == active:
                prob[s] = 0
            else:
                temp = fact * (np.abs(cupd.coeff.out[s])**2 - np.abs(cupd.coeff.inp[s])**2)
                prob[s] = max(0, temp)

        self.prob.out = prob
        self.hop.out = self._check_hop(prob, active, dt)

class MASHChecker(HoppingUpdater):
    key = "mash"

    def update(self, mols: list[Molecule], dt: float, active: int):
        nst = mols[-1].n_states
        prob = self.prob.inp
        for s in range(nst):
            if s == active:
                continue
            if mols[-1].bloch_n3[s,2] < 0:
                prob[s] = 1
        self.prob.out = prob
        self.hop.out = self._check_hop(prob, active, dt)

class MISHChecker(HoppingUpdater):
    key = "mish"

    def update(self, mols: list[Molecule], dt: float, active: int):
        prob = self.prob.inp
        prob[:] = 0
        target = np.argmax(np.abs(mols[-1].coeff_s)**2)
        prob[target] = 1.
        self.prob.out = prob
        self.hop.out = self._check_hop(prob, active, dt)

# class FSSHCChecker(HoppingUpdater, ):
#     key = "fssh-c"
#
#     def __init__(self, *, seed = None, **config):
#         super().__init__(**config)
#         self.r = 0.
#     def _check_hop(self, prob: np.ndarray, active: int):
#         return
#
