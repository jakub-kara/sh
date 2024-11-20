import numpy as np
from classes.meta import Factory
from classes.molecule import Molecule
from classes.constants import Constants
from classes.out import Output
from updaters.nuclear import NuclearUpdater
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater
from electronic.electronic import ESTProgram

# TODO: move Control to Dynamics
class Dynamics(metaclass = Factory):
    mode = ""

    def __init__(self, *, dynamics: dict, **config: dict):
        tconv = {
            "fs": 1/Constants.au2fs,
            "au": 1,
        }[dynamics.get("tunit", "au")]

        self._dt = dynamics["dt"] * tconv
        self._dtmax = self._dt
        self._end = dynamics["tmax"] * tconv
        self._time = 0
        self._step = 0
        self._enthresh = dynamics.get("enthresh", 1000)

        self._stepok = True


    @property
    def is_finished(self):
        return self._time > self._end

    @property
    def dt(self):
        return self._dt

    @property
    def curr_step(self):
        return self._step

    @property
    def curr_time(self):
        return self._time

    @property
    def en_thresh(self):
        return self.en_thresh

    @property
    def step_ok(self):
        return self._stepok

    def next_step(self):
        self._time += self.dt
        self._step += 1

    def check_energy(self, energy_diff):
        self._stepok = energy_diff < self._enthresh

    def calculate_acceleration(self, mol: Molecule):
        raise NotImplementedError

    def potential_energy(self, mol: Molecule):
        raise NotImplementedError

    def prepare_traj(self, mol: Molecule):
        out = Output()
        out.open_log()
        est = ESTProgram()
        self.setup_est(mode = self.mode)
        est.run(mol)
        est.read(mol, mol)
        self.calculate_acceleration(mol)
        est.reset_calc()

    # might have a better name
    def adjust_nuclear(self, mol: Molecule):
        raise NotImplementedError

    def get_mode(self):
        return self.mode + TDCUpdater().mode + CoeffUpdater().mode

    def setup_est(self, mode: str = ""):
        pass

    def update_nuclear(self, mols: list[Molecule]):
        nupd = NuclearUpdater()
        return nupd.update(mols, self.dt, self)

    def update_quantum(self, mols: list[Molecule]):
        self.update_tdc(mols)
        self.update_coeff(mols)

    def update_tdc(self, mols: list[Molecule]):
        tdcupd = TDCUpdater()
        tdcupd.elapsed(self.curr_step)
        tdcupd.run(mols, self.dt)
        mols[-1].nacdt_ss = tdcupd.tdc.out

    def update_coeff(self, mols: list[Molecule]):
        cupd = CoeffUpdater()
        cupd.elapsed(self.curr_step)
        cupd.run(mols, self.dt)
        mols[-1].coeff_s = cupd.coeff.out

    def _get_eff_nac(self, mol: Molecule):
        nac_eff = np.zeros_like(mol.nacdr_ssad)
        for i in range(mol.n_states):
            for j in range(i):
                diff = mol.grad_sad[i] - mol.grad_sad[j]
                alpha = (mol.nacdt_ss[i,j] - np.sum(diff * mol.vel_ad)) / np.sum(mol.vel_ad**2)
                nac_eff[i,j] = diff + alpha * mol.vel_ad
                nac_eff[j,i] = -nac_eff[i,j]
        return nac_eff

    def dat_header(self, dic: dict, record: list):
        return dic

    def dat_dict(self, dic: dict, record: list):
        return dic