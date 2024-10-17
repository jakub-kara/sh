import numpy as np
from scipy.linalg import expm
from classes.molecule import Molecule
from integrators.tdc import TDCUpdater
from integrators import select_tdcupd

class CoeffUpdater:
    def __init__(self, n_qsteps: int, *args, **kwargs):
        self._nqsteps = n_qsteps

    def is_ready(self, step: int):
        return True

    def update(self, mol: list[Molecule], tdcupd: TDCUpdater, dt: float, *args, **kwargs):
        for i in range(self._nqsteps):
            frac = (i + 0.5) / self._nqsteps
            ham = frac * mol[-1].pes.ham_eig_ss + (1 - frac) * mol[-2].pes.ham_eig_ss
            arg = -(1.j * ham + tdcupd.interpolate(mol, frac)) * dt / self._nqsteps
            mol[-1].pes.coeff_s = expm(arg) @ mol[-1].pes.coeff_s
            yield frac

class LDCoeffUpdater(CoeffUpdater):
    def __init__(self, *args, **kwargs):
        self._nqsteps = 1

    def is_ready(self, step):
        return step >= 2

    def update(self, mol: list[Molecule], tdcupd: TDCUpdater, *args, **kwargs):
        mol[-1].pes.coeff_s = tdcupd.rmat @ mol[-1].pes.coeff_s
        yield 1/2

class QuantumUpdater:
    def __init__(self, *, n_qsteps: int, tdcupd: str, **config):
        self._nqsteps = n_qsteps
        self._tdcupd: TDCUpdater = select_tdcupd(tdcupd)(self._nqsteps)
        self._cupd: CoeffUpdater = self._select_cupd(tdcupd)(self._nqsteps)

    @property
    def mode(self):
        return self._tdcupd.mode

    def _select_cupd(self, key):
        res = "propmat"
        if key == "ld":
            res = "rmat"
        items = {
            "propmat": CoeffUpdater,
            "rmat": LDCoeffUpdater,
        }
        return items[res]

    def update(self, mol: list[Molecule], dt: float, step: int, *args, **kwargs):
        self._tdcupd.fix_phase(mol[-1])

        if self._tdcupd.is_ready(step):
            self._tdcupd.update(mol, dt)
        else:
            self._tdcupd.no_update(mol)

        mol[-1].pes.coeff_s = mol[-2].pes.coeff_s
        if self._cupd.is_ready(step):
            mol[-1].pes.coeff_s = mol[-1].pes.trans_ss @ mol[-1].pes.coeff_s
            self._update_coeff(mol, dt)
            mol[-1].pes.coeff_s = mol[-1].pes.trans_ss.conj().T @ mol[-1].pes.coeff_s

    def _update_coeff(self, mol: list[Molecule], dt):
        for frac in self._cupd.update(mol, self._tdcupd, dt):
            pass