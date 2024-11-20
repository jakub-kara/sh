import numpy as np
from typing import Callable
from .nuclear import NuclearUpdater
from classes.molecule import Molecule

class Yoshida4(NuclearUpdater, key = "y4"):
    w0 = -np.cbrt(2) / (2 - np.cbrt(2))
    w1 = 1 / (2 - np.cbrt(2))
    c = np.array([w1/2, (w0+w1)/2, (w0+w1)/2, w1/2])
    d = np.array([w1, w0, w1])

    def update(self, mols: list[Molecule], dt: float, fun: Callable):
        mol = mols[-1]
        temp = mol.copy_all()
        dt = dt / 4

        # x1, a1, v1
        temp.pos_ad += self.c[0] * temp.vel_ad * dt
        fun(temp)
        temp.vel_ad += self.d[0] * temp.acc_ad * dt

        # x2, a2, v2
        temp.pos_ad += self.c[1] * temp.vel_ad * dt
        fun(temp)
        temp.vel_ad += self.d[1] * temp.acc_ad * dt

        # x3, a3, v3
        temp.pos_ad += self.c[2] * temp.vel_ad * dt
        fun(temp)
        temp.vel_ad += self.d[2] * temp.acc_ad * dt

        temp.pos_ad += self.c[3] * temp.vel_ad * dt
        return temp