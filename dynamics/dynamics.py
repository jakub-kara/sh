import numpy as np
from copy import deepcopy
from classes.meta import Factory
from classes.molecule import Molecule
from classes.out import Output
from updaters.composite import CompositeIntegrator
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater
from electronic.electronic import ESTProgram

class Dynamics(metaclass = Factory):
    mode = ""

    def __init__(self, *, dynamics: dict, **config: dict):
        self.split = None

    def calculate_acceleration(self, mol: Molecule):
        raise NotImplementedError

    def potential_energy(self, mol: Molecule):
        raise NotImplementedError

    def total_energy(self, mol: Molecule):
        return self.potential_energy(mol) + mol.kinetic_energy

    def population(self, mol: Molecule, s: int):
        return np.abs(mol.coeff_s[s])**2

    def read_coeff(self, mol: Molecule, file = None):
        if file is None:
            return
        data = np.genfromtxt(file)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape != (mol.n_states, 2):
            raise ValueError(f"Invalid coeff input format in {file}")
        mol.coeff_s[:] = data[:,0]
        mol.coeff_s += 1j*data[:,1]

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
    def adjust_nuclear(self, mol: Molecule, dt: float):
        raise NotImplementedError

    def get_mode(self):
        return self.mode + TDCUpdater().mode + CoeffUpdater().mode

    def setup_est(self, mode: str = ""):
        pass

    def steps_elapsed(self, steps: int):
        TDCUpdater().elapsed(steps)
        CoeffUpdater().elapsed(steps)

    def update_nuclear(self, mols: list[Molecule], dt: float):
        nupd = CompositeIntegrator()
        return nupd.run(mols, dt, self)
        # return nupd.update(mols, dt, self)

    def update_quantum(self, mols: list[Molecule], dt: float):
        self.update_tdc(mols, dt)
        self.update_coeff(mols, dt)

    def update_tdc(self, mols: list[Molecule], dt: float):
        tdcupd = TDCUpdater()
        tdcupd.run(mols, dt)
        mols[-1].nacdt_ss = tdcupd.tdc.out

    def update_coeff(self, mols: list[Molecule], dt: float):
        cupd = CoeffUpdater()
        cupd.run(mols, dt)
        mols[-1].coeff_s = cupd.coeff.out

    def _eff_nac(self, mol: Molecule):
        nac_eff = np.zeros_like(mol.nacdr_ssad)
        for i in range(mol.n_states):
            for j in range(i):
                diff = mol.grad_sad[i] - mol.grad_sad[j]
                if np.abs(mol.nacdt_ss[i,j]) < 1e-8:
                    alpha = 0
                else:
                    alpha = (mol.nacdt_ss[i,j] - np.sum(diff * mol.vel_ad)) / np.sum(mol.vel_ad**2)
                nac_eff[i,j] = diff + alpha * mol.vel_ad
                nac_eff[j,i] = -nac_eff[i,j]
        return nac_eff

    def split_mol(self, mol: Molecule):
        out1 = deepcopy(mol)
        out1.coeff_s[:] = 0
        out1.coeff_s[self.split] = mol.coeff_s[self.split]
        out1.coeff_s /= np.sqrt(np.sum(np.abs(out1.coeff_s)**2))

        out2 = deepcopy(mol)
        out2.coeff_s[self.split] = 0
        out2.coeff_s /= np.sqrt(np.sum(np.abs(out2.coeff_s)**2))
        return out1, out2

    def dat_header(self, dic: dict, record: list):
        return dic

    def dat_dict(self, dic: dict, record: list):
        return dic

    def h5_dict(self):
        return {}