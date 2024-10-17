import numpy as np
from typing import Callable
from integrators.integrators import Integrator2
from classes.molecule import Molecule

class VelocityVerlet(Integrator2):
    def __init__(self):
        super().__init__()
        self.name = "vv"
        self.order = 2

    def integrate(self, fun: Callable, dt: float, *args: Molecule):
        mol = args[-1]
        temp = mol.copy_all()
        temp.pos_ad = mol.pos_ad + dt*mol.vel_ad + 0.5*dt**2*mol.acc_ad
        fun(temp)
        temp.vel_ad = mol.vel_ad + 0.5 * dt * (mol.acc_ad + temp.acc_ad)

        return temp