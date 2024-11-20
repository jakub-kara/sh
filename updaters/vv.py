import numpy as np
from typing import Callable
from .nuclear import NuclearUpdater
from classes.molecule import Molecule
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram

class VelocityVerlet(NuclearUpdater, key = "vv"):
    def update(self, mols: list[Molecule], dt: float, dyn: Dynamics):
        # update position
        mol = mols[-1]
        temp = mol.copy_all()
        temp.pos_ad = mol.pos_ad + dt * mol.vel_ad + 0.5 * dt**2 * mol.acc_ad

        # calculate est
        est = ESTProgram()
        dyn.setup_est(mode = dyn.get_mode())
        est.run(temp)
        est.read(temp, ref = mol)
        dyn.calculate_acceleration(temp)
        est.reset_calc()

        # update velocity
        temp.vel_ad = mol.vel_ad + 0.5 * dt * (mol.acc_ad + temp.acc_ad)

        return temp