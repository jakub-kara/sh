import numpy as np
from .nuclear import NuclearUpdater
from classes.molecule import Molecule
from dynamics.base import Dynamics

class VelocityVerlet(NuclearUpdater):
    key = "vv"

    def update(self, mols: list[Molecule], dt: float):
        # update position
        dyn = Dynamics()
        mol = mols[-1]
        out: Molecule = self.out.inp.copy_all()
        out.pos_ad = mol.pos_ad + dt * mol.vel_ad + 0.5 * dt**2 * mol.acc_ad

        dyn.run_est(out, ref = mol, mode = dyn.step_mode(out))
        dyn.update_quantum(mols + [out], dt)
        dyn.calculate_acceleration(out)
        out.vel_ad = mol.vel_ad + 0.5 * dt * (mol.acc_ad + out.acc_ad)

        self.out.out = out