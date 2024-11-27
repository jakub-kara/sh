import numpy as np
from .nuclear import NuclearUpdater
from .coeff import CoeffUpdater
from classes.molecule import Molecule
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram

class VelocityVerlet(NuclearUpdater, key = "vv"):
    def update(self, mols: list[Molecule], dt: float, dyn: Dynamics):
        # update position
        mol = mols[-1]
        out = mol.copy_all()
        out.pos_ad = mol.pos_ad + dt * mol.vel_ad + 0.5 * dt**2 * mol.acc_ad

        # calculate est
        est = ESTProgram()
        dyn.setup_est(mode = dyn.get_mode())
        est.run(out)
        est.read(out, ref = mol)
        est.reset_calc()

        dyn.update_quantum(mols + [out], dt)
        dyn.calculate_acceleration(out)
        out.vel_ad = mol.vel_ad + 0.5 * dt * (mol.acc_ad + out.acc_ad)
        return out