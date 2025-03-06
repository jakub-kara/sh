import numpy as np
from .sh import SurfaceHopping
from .checker import HoppingUpdater
from classes.molecule import Molecule
from classes.timestep import Timestep
from electronic.electronic import ESTProgram
from updaters.composite import CompositeIntegrator
from updaters.coeff import BlochUpdater

class MASH(SurfaceHopping):
    ''' Lawrence and Richardson's "unSMASH". Reduces to Mannouch and Richardson's "MASH" for two state case '''
    key = "mash"

    def __init__(self, **config):
        config["nuclear"]["mixins"].append("bloch")
        super().__init__(**config)
        BlochUpdater(**config["quantum"])
        HoppingUpdater["mash"](**config["quantum"])

        self._rescale = "nac"
        self._reverse = True

    def read_coeff(self, mol: Molecule, file=None):
        if file is None:
            mol.bloch_n3[:, 2] = 1
            mol.bloch_n3[mol.active, :] = None
            return
        data = np.genfromtxt(file)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape != (mol.n_states - 1, 3):
            raise ValueError(f"Invalid bloch input format in {file}")
        mol.bloch_n3[:mol.active] = data[:mol.active]
        mol.bloch_n3[mol.active + 1:] = data[mol.active:]

    def adjust_nuclear(self, mols: list[Molecule], dt: float):
        mol = mols[-1]
        self.update_target(mols, dt)

        print(mol.bloch_n3)
        print(f"target: {mol.target} \t\tactive: {mol.active}")

        if mol.hop_ready():
            if self._has_energy(mol, self._get_delta(mol)):
                self._adjust_velocity(mol, self._get_delta(mol))
                self._swap_bloch(mol)
                CompositeIntegrator().to_init()
                mol.hop()

                self.setup_est(mol, mode = "a")
                est = ESTProgram()
                est.run(mol)
                est.read(mol)
                self.calculate_acceleration(mol)
                est.reset_calc()
            else:
                self._reverse_velocity(mol, self._get_delta(mol))
                self._reverse_bloch(mol)
                mol.nohop()

    def steps_elapsed(self, steps):
        super().steps_elapsed(steps)
        BlochUpdater().elapsed(steps)

    def update_quantum(self, mols, dt: float):
        self.update_tdc(mols, dt)
        # self.update_coeff(mols, dt)
        self.update_bloch(mols, dt)

    def update_bloch(self, mols: list[Molecule], dt: float):
        bupd = BlochUpdater()
        bupd.run(mols, dt)
        mols[-1].bloch_n3 = bupd.bloch.out

    # def _has_energy(self, mol: Molecule):
    #     d2 = np.sum(mol.nacdr_ssad[self.target, self.active]**2)
    #     pmw = mol.vel_ad * np.sqrt(mol.mass_a[:,None])
    #     ppar = np.sum(pmw * mol.nacdr_ssad[self.target, self.active]) / d2 * mol.nacdr_ssad[self.target, self.active]
    #     return np.sum(ppar**2) / 2 + mol.ham_eig_ss[self.active, self.active] - mol.ham_eig_ss[self.target, self.target] >= 0

    # def _adjust_velocity(self, mol: Molecule):
    #     d2 = np.sum(mol.nacdr_ssad[self.target, self.active]**2)
    #     pmw = mol.vel_ad * np.sqrt(mol.mass_a[:,None])
    #     ppar = np.sum(pmw * mol.nacdr_ssad[self.target, self.active]) / d2 * mol.nacdr_ssad[self.target, self.active]
    #     pperp = pmw - ppar
    #     pfin = np.sqrt(1 + 2 * (mol.ham_eig_ss[self.active, self.active] - mol.ham_eig_ss[self.target, self.target]) / np.sum(ppar**2)) * ppar
    #     mol.vel_ad = (pperp + pfin) / np.sqrt(mol.mass_a[:,None])

    # def _reverse_velocity(self, mol: Molecule):
    #     d2 = np.sum(mol.nacdr_ssad[self.target, self.active]**2)
    #     pmw = mol.vel_ad * np.sqrt(mol.mass_a[:,None])
    #     ppar = np.sum(pmw * mol.nacdr_ssad[self.target, self.active]) / d2 * mol.nacdr_ssad[self.target, self.active]
    #     pperp = pmw - ppar
    #     pfin = -ppar
    #     mol.vel_ad = (pperp + pfin) / np.sqrt(mol.mass_a[:,None])

    def _swap_bloch(self, mol: Molecule):
        swp = np.array([1, -1, -1])
        for s in range(mol.n_states):
            if s == self.active:
                mol.bloch_n3[s] = mol.bloch_n3[self._target] * swp
                mol.bloch_n3[self._target] = None

    def _reverse_bloch(self, mol: Molecule):
        mol.bloch_n3[self._target, 2] *= -1
