import numpy as np
from .sh import SurfaceHopping
from .checker import HoppingUpdater
from classes.molecule import MoleculeBloch
from electronic.electronic import ESTProgram
from updaters.coeff import BlochUpdater
from updaters.tdc import TDCUpdater

class unSMASH(SurfaceHopping, key = "unsmash"):
    ''' Lawrence and Richardson's "unSMASH". Reduces to Mannouch and Richardson's "MASH" for two state case '''
    def __init__(self, **config):
        config["nuclear"]["pes"] = "bloch"
        config["quantum"]["coeff_upd"] = "bloch"
        super().__init__(**config)
        HoppingUpdater(key = "mash", **config["quantum"])

    def adjust_nuclear(self, mols: list[MoleculeBloch]):
        mol = mols[-1]
        self.update_target(mols, self.dt)

        print(mol.bloch_n3)
        print(f"target: {self.target} \t\tactive: {self.active}")

        if self.hop_ready():
            if "n" not in self.mode:
                self.setup_est(mode = "n")
                est = ESTProgram()
                est.run(mol)
                est.read(mol, mol)
                est.reset_calc()

            if self._has_energy(mol):
                self._adjust_velocity(mol)
                self._swap_bloch(mol)
                self._hop()

                self.setup_est(mode = "a")
                est = ESTProgram()
                est.run(mol)
                est.read(mol)
                self.calculate_acceleration(mol)
                est.reset_calc()
            else:
                self._reverse_velocity(mol)
                self._reverse_bloch(mol)
                self._nohop()

    def prepare_traj(self, mol: MoleculeBloch):
        mol.bloch_n3[:, 2] = 1
        mol.bloch_n3[self.active, :] = None
        super().prepare_traj(mol)

    def update_quantum(self, mols, dt: float):
        self.update_tdc(mols, dt)
        self.update_bloch(mols, dt)

    def update_bloch(self, mols: list[MoleculeBloch], dt: float):
        bupd = BlochUpdater()
        bupd.elapsed(self.curr_step)
        bupd.run(mols, dt, self.active)
        mols[-1].bloch_n3 = bupd.bloch.out

    def _has_energy(self, mol: MoleculeBloch):
        d2 = np.sum(mol.nacdr_ssad[self.target, self.active]**2)
        pmw = mol.vel_ad * np.sqrt(mol.mass_a[:,None])
        ppar = np.sum(pmw * mol.nacdr_ssad[self.target, self.active]) / d2 * mol.nacdr_ssad[self.target, self.active]
        return np.sum(ppar**2) / 2 + mol.ham_eig_ss[self.active, self.active] - mol.ham_eig_ss[self.target, self.target] >= 0

    def _adjust_velocity(self, mol: MoleculeBloch):
        d2 = np.sum(mol.nacdr_ssad[self.target, self.active]**2)
        pmw = mol.vel_ad * np.sqrt(mol.mass_a[:,None])
        ppar = np.sum(pmw * mol.nacdr_ssad[self.target, self.active]) / d2 * mol.nacdr_ssad[self.target, self.active]
        pperp = pmw - ppar
        pfin = np.sqrt(1 + 2 * (mol.ham_eig_ss[self.active, self.active] - mol.ham_eig_ss[self.target, self.target]) / np.sum(ppar**2)) * ppar
        mol.vel_ad = (pperp + pfin) / np.sqrt(mol.mass_a[:,None])

    def _reverse_velocity(self, mol: MoleculeBloch):
        d2 = np.sum(mol.nacdr_ssad[self.target, self.active]**2)
        pmw = mol.vel_ad * np.sqrt(mol.mass_a[:,None])
        ppar = np.sum(pmw * mol.nacdr_ssad[self.target, self.active]) / d2 * mol.nacdr_ssad[self.target, self.active]
        pperp = pmw - ppar
        pfin = -ppar
        mol.vel_ad = (pperp + pfin) / np.sqrt(mol.mass_a[:,None])

    def _swap_bloch(self, mol: MoleculeBloch):
        swp = np.array([1, -1, -1])
        for s in range(mol.n_states):
            if s == self.active:
                mol.bloch_n3[s] = mol.bloch_n3[self._target] * swp
                mol.bloch_n3[self._target] = None

    def _reverse_bloch(self, mol: MoleculeBloch):
        mol.bloch_n3[self._target, 2] *= -1