import numpy as np
from .sh import SurfaceHopping
from .checker import HoppingUpdater
from classes.molecule import Molecule
from classes.timestep import Timestep
from electronic.base import ESTProgram
from updaters.composite import CompositeIntegrator
from updaters.coeff import BlochUpdater

class MASH(SurfaceHopping):
    ''' Lawrence and Richardson's "unSMASH". Reduces to Mannouch and Richardson's "MASH" for two state case '''
    key = "mash"

    def __init__(self, **config):
        config["nuclear"]["mixins"] = "bloch"
        super().__init__(**config)
        BlochUpdater(**config["quantum"])
        HoppingUpdater["mash"](**config["quantum"])

        self._rescale = "nac"
        self._reverse = True

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

                est = ESTProgram()
                est.request(*self.mode(mol))
                est.run(mol)
                est.read(mol, ref = mols[-2])
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

    def _swap_bloch(self, mol: Molecule):
        swp = np.array([1, -1, -1])
        for s in range(mol.n_states):
            if s == mol.active:
                mol.bloch_n3[s] = mol.bloch_n3[mol.target] * swp
                mol.bloch_n3[mol.target] = None

    def _reverse_bloch(self, mol: Molecule):
        mol.bloch_n3[mol.target, 2] *= -1
