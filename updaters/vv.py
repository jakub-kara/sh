import numpy as np
from .nuclear import NuclearUpdater
from classes.molecule import Molecule
from classes.timestep import Timestep
from dynamics.base import Dynamics

class VelocityVerlet(NuclearUpdater):
    key = "vv"

    # TODO: implement leapfrog with adaptive timestep
    def update(self, mols: list[Molecule], ts: Timestep):
        # update position
        dyn = Dynamics()
        mol = mols[-1]
        out: Molecule = self.out.inp.copy_all()
        out.pos_ad = mol.pos_ad + ts.dt * mol.vel_ad + 0.5 * ts.dt**2 * mol.acc_ad

        dyn.run_est(out, ref = mol, mode = dyn.step_mode(out))
        dyn.update_quantum(mols + [out], ts)
        dyn.calculate_acceleration(out)
        out.vel_ad = mol.vel_ad + 0.5 * ts.dt * (mol.acc_ad + out.acc_ad)

        self.out.out = out