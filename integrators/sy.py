import numpy as np
from typing import Callable
from integrators.integrators import Integrator2
from integrators.am import *

class SYBase(Integrator2):
    def __init__(self):
        super().__init__()
        self.substeps = 1
        self.a = np.empty(1)
        self.b = np.empty(1)
        self.c = 1

    def integrate(self, fun: Callable, dt: float, *args: Molecule):
        temp = args[-1].copy_all()
        # find new position as a weighted sum of previous positions and accelerations
        temp.pos_ad = -np.einsum("j,j...->...", self.a[:-1], np.array([mol.pos_ad for mol in args])) + dt**2*np.einsum("j,j...->...", self.b[:-1], np.array([mol.vel_ad for mol in args]))
        temp.pos_ad /= self.a[-1]

        return temp

class SY2(SYBase):
    def __init__(self):
        super().__init__()
        self.name = "sy2"
        self.steps = 2
        self.order = 2
        self.a = np.array([1, -2, 1])
        self.b = np.array([0, 1, 0])

class SY4(SYBase):
    def __init__(self):
        super().__init__()
        self.name = "sy4"
        self.steps = 4
        self.order = 4
        self.a = np.array([1, -1, 0, -1, 1])
        self.b = np.array([0, 5/4, 1/2, 5/4, 0])

class SY6(SYBase):
    def __init__(self):
        super().__init__()
        self.name = "sy6"
        self.steps = 6
        self.order = 6
        self.a = np.array([1, -2, 2, -2, 2, -2, 1])
        self.b = np.array([0, 317/240, -31/30, 291/120, -31/30, 317/240, 0])
        
class SY8(SYBase):
    def __init__(self):
        super().__init__()
        self.name = "sy8"
        self.steps = 8
        self.order = 8
        self.a = np.array([1, -2, 2, -1, 0, -1, 2, -2, 1])
        self.b = np.array([0, 17671, -23622, 61449, -50516, 61449, -23622, 17671, 0])/12096

class SY8b(SYBase):
    def __init__(self):
        super().__init__()
        self.name = "sy8b"
        self.steps = 8
        self.order = 8
        self.a = np.array([1, 0, 0, -1/2, -1, -1/2, 0, 0, 1])
        self.b = np.array([0, 192481, 6582, 816783, -156812, 816783, 6582, 192481, 0])/120960

class SY8c(SYBase):
    def __init__(self):
        super().__init__()
        self.name = "sy8c"
        self.steps = 8
        self.order = 8
        self.a = np.array([1, -1, 0, 0, 0, 0, 0, -1, 1])
        self.b = np.array([0, 13207, -8934, 42873, -33812, 42873, -8934, 13207, 0])/8640

class SYAMBase(Integrator2):
    def __init__(self):
        super().__init__()
        self.substeps = 1
        self.sy: SYBase = None
        self.am: AMBase = None

    def integrate(self, fun: Callable, dt: float, *args: Molecule):
        temp = args[-1].copy_all()
        mols = args[-self.steps:]
        # find new position as a weighted sum of previous positions and accelerations
        temp.pos_ad = -np.einsum("j,j...->...", self.sy.a[:-1], np.array([mol.pos_ad for mol in mols])) + dt**2*np.einsum("j,j...->...", self.sy.b[:-1], np.array([mol.acc_ad for mol in mols]))
        temp.pos_ad /= self.sy.a[-1]
        # calculate new acceleration
        fun(temp)
        # calculate new velocity from new acceleration, previous velocities, and previous accelerations
        temp.vel_ad += dt*np.einsum("j,j...->...", self.am.b[:-1], np.array([mol.acc_ad for mol in mols])[1:]) + dt*self.am.b[-1]*temp.acc_ad

        return temp
    
class SYAM4(SYAMBase):
    def __init__(self):
        super().__init__()
        self.name = "syam4"
        self.steps = 4
        self.order = 4
        self.sy = SY4()
        self.am = AM4()

class SYAM6(SYAMBase):
    def __init__(self):
        super().__init__()
        self.name = "syam6"
        self.steps = 6
        self.order = 6
        self.sy = SY6()
        self.am = AM6()

class SYAM8(SYAMBase):
    def __init__(self):
        super().__init__()
        self.name = "syam8"
        self.steps = 8
        self.order = 8
        self.sy = SY8()
        self.an = AM8()