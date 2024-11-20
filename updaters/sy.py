import numpy as np
from typing import Callable
from .nuclear import NuclearUpdater
from .am import *

class SYBase(NuclearUpdater):
    substeps = 1
    a = np.empty(1)
    b = np.empty(1)
    c = 1

    def update(self, mols: list[Molecule], dt: float, fun: Callable):
        # find new position as a weighted sum of previous positions and accelerations
        pos_ad = -np.einsum("j,j...->...", self.a[:-1], np.array([mol.pos_ad for mol in mols])) + dt**2*np.einsum("j,j...->...", self.b[:-1], np.array([mol.vel_ad for mol in mols]))
        pos_ad /= self.a[-1]

        return (pos_ad, mols[-1].vel_ad.copy(), mols[-1].acc_ad.copy())

class SY2(SYBase, key = "sy2"):
    steps = 2
    a = np.array([1, -2, 1])
    b = np.array([0, 1, 0])

class SY4(SYBase, key = "sy4"):
    steps = 4
    a = np.array([1, -1, 0, -1, 1])
    b = np.array([0, 5/4, 1/2, 5/4, 0])

class SY6(SYBase, key = "sy6"):
    name = "sy6"
    steps = 6
    a = np.array([1, -2, 2, -2, 2, -2, 1])
    b = np.array([0, 317/240, -31/30, 291/120, -31/30, 317/240, 0])

class SY8(SYBase, key = "sy8"):
    steps = 8
    a = np.array([1, -2, 2, -1, 0, -1, 2, -2, 1])
    b = np.array([0, 17671, -23622, 61449, -50516, 61449, -23622, 17671, 0])/12096

class SY8b(SYBase, key = "sy8b"):
    steps = 8
    a = np.array([1, 0, 0, -1/2, -1, -1/2, 0, 0, 1])
    b = np.array([0, 192481, 6582, 816783, -156812, 816783, 6582, 192481, 0])/120960

class SY8c(SYBase, key = "sy8c"):
    steps = 8
    a = np.array([1, -1, 0, 0, 0, 0, 0, -1, 1])
    b = np.array([0, 13207, -8934, 42873, -33812, 42873, -8934, 13207, 0])/8640

class SYAMBase(NuclearUpdater):
    substeps = 1
    sy: SYBase = None
    am: AMBase = None

    def update(self, mols: list[Molecule], dt: float, fun: Callable):
        temp = mols[-1].copy_all()
        # find new position as a weighted sum of previous positions and accelerations
        temp.pos_ad = -np.einsum("j,j...->...", self.sy.a[:-1], np.array([mol.pos_ad for mol in mols])) + dt**2*np.einsum("j,j...->...", self.sy.b[:-1], np.array([mol.acc_ad for mol in mols]))
        temp.pos_ad /= self.sy.a[-1]
        # calculate new acceleration
        fun(temp)
        # calculate new velocity from new acceleration, previous velocities, and previous accelerations
        temp.vel_ad += dt*np.einsum("j,j...->...", self.am.b[:-1], np.array([mol.acc_ad for mol in mols])[1:]) + dt*self.am.b[-1]*temp.acc_ad

        return temp

class SYAM4(SYAMBase, key = "syam4"):
    steps = 4
    sy = SY4
    am = AM4

class SYAM6(SYAMBase, key = "syam6"):
    steps = 6
    sy = SY6
    am = AM6

class SYAM8(SYAMBase, key = "syam8"):
    steps = 8
    sy = SY8
    am = AM8