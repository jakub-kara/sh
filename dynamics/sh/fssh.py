import numpy as np
from .sh import SurfaceHopping
from .checker import HoppingUpdater
from classes.molecule import Molecule
from classes.out import Output
from electronic.base import ESTMode
from updaters.composite import CompositeIntegrator

class FSSH(SurfaceHopping):
    key = "fssh"

    def __init__(self, *, dynamics, **config):
        config["nuclear"]["mixins"] = "sh"
        super().__init__(dynamics=dynamics, **config)
        HoppingUpdater[dynamics.get("prob", "tdc")](**dynamics, **config["quantum"])

    def adjust_nuclear(self, mols: list[Molecule], dt: float):
        out = Output()
        mol = mols[-1]
        self._decoherence(mol, dt)
        self.update_target(mols, dt)

        out.write_log(f"target: {mol.target} \t\tactive: {mol.active}")
        # print(f"Final pops: {np.abs(mol.coeff_s)**2}")
        # print(f"Check sum:  {np.sum(np.abs(mol.coeff_s)**2)}")
        if mol.hop_ready():
            delta = self._get_delta(mol)
            if self._has_energy(mol, delta):
                out.write_log("Hop succesful")
                self._adjust_velocity(mol, delta)
                mol.hop()
                CompositeIntegrator().to_init()
                out.write_log(f"New state: {mol.active}")

                self.run_est(mol, mols[-2], ESTMode("a")(mol))
                self.calculate_acceleration(mol)
            else:
                out.write_log("Hop failed")
                # out.write_log(f'vel: \n{mol.vel_ad}')
                # out.write_log(f'delta: \n{delta}')
                out.write_log(f'available kinetic energy = {self._avail_kinetic_energy(mol,delta)}')
                out.write_log(f'energy difference {mol.ham_eig_ss[mol.active, mol.active] - mol.ham_eig_ss[mol.target, mol.target]}')
                if self._reverse:
                    out.write_log("Reversing velocity")
                    self._reverse_velocity(mol, delta)
                mol.nohop()
            out.write_log()
