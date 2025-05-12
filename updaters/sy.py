import numpy as np
from .nuclear import NuclearUpdater
from .am import *
from classes.timestep import Timestep
from dynamics.base import Dynamics
from electronic.base import ESTProgram

class SYBase(NuclearUpdater):
    substeps = 1
    a = np.empty(1)
    b = np.empty(1)
    c = 1

    def update(self, mols: list[Molecule], ts: Timestep):
        raise NotImplementedError
        # find new position as a weighted sum of previous positions and accelerations
        pos_ad = -np.einsum("j,j...->...", self.a[:-1], np.array([mol.pos_ad for mol in mols])) + dt**2*np.einsum("j,j...->...", self.b[:-1], np.array([mol.vel_ad for mol in mols]))
        pos_ad /= self.a[-1]

        return (pos_ad, mols[-1].vel_ad.copy(), mols[-1].acc_ad.copy())

class SY2(SYBase):
    key = "sy2"
    steps = 2
    a = np.array([1, -2, 1])
    b = np.array([0, 1, 0])

class SY4(SYBase):
    key = "sy4"
    steps = 4
    a = np.array([1, -1, 0, -1, 1])
    b = np.array([0, 5/4, 1/2, 5/4, 0])

class SY6(SYBase):
    key = "sy6"
    steps = 6
    a = np.array([1, -2, 2, -2, 2, -2, 1])
    b = np.array([0, 317/240, -31/30, 291/120, -31/30, 317/240, 0])

class SY8(SYBase):
    key = "sy8"
    steps = 8
    a = np.array([1, -2, 2, -1, 0, -1, 2, -2, 1])
    b = np.array([0, 17671, -23622, 61449, -50516, 61449, -23622, 17671, 0])/12096

class SY8b(SYBase):
    key = "sy8b"
    steps = 8
    a = np.array([1, 0, 0, -1/2, -1, -1/2, 0, 0, 1])
    b = np.array([0, 192481, 6582, 816783, -156812, 816783, 6582, 192481, 0])/120960

class SY8c(SYBase):
    key = "sy8c"
    steps = 8
    a = np.array([1, -1, 0, 0, 0, 0, 0, -1, 1])
    b = np.array([0, 13207, -8934, 42873, -33812, 42873, -8934, 13207, 0])/8640

class SYAMBase(NuclearUpdater):
    substeps = 1
    sy: SYBase = None
    am: AMBase = None

    def update(self, mols: list[Molecule], ts: Timestep):
        dyn = Dynamics()
        out = mols[-1].copy_all()
        temp = mols[-self.steps:]
        # find new position as a weighted sum of previous positions and accelerations
        out.pos_ad = -np.einsum("j,j...->...", self.sy.a[:-1], np.array([mol.pos_ad for mol in temp])) + ts.dt**2*np.einsum("j,j...->...", self.sy.b[:-1], np.array([mol.acc_ad for mol in temp]))
        out.pos_ad /= self.sy.a[-1]

        # calculate new acceleration
        dyn.run_est(out, ref = mols[-1], mode = dyn.step_mode(out))
        dyn.update_quantum(mols + [out], ts)
        dyn.calculate_acceleration(out)

        # calculate new velocity from new acceleration, previous velocities, and previous accelerations
        out.vel_ad += ts.dt*np.einsum("j,j...->...", self.am.b[:-1], np.array([mol.acc_ad for mol in temp])[1:]) + ts.dt*self.am.b[-1]*out.acc_ad

        self.out.out = out

def am4_coeffs(h0, h1, h2):
    t1 = h0
    t2 = h0+h1
    t3 = h0+h1+h2

    b0 = -(t2-t3)**3 * (2*t1-t2-t3)/((-t1)*(-t2)*(-t3))/12
    b1 = (t2-t3)**3 * (t2+t3)/(t1*(t1-t2)*(t1-t3))/12
    b2 = -(t3-t2)**2 * (t3**2 + 2*t2*t3 + 3*t2**2 - 2*t1*(t3+2*t2))/(t2*(t2-t1)*(t2-t3))/12
    b3 = (t2-t3)**2 * (t2**2 + 2*t2*t3 + 3*t3**2 - 2*t1*(t2+2*t3))/(t3*(t3-t1)*(t3-t2))/12

    return np.array([b3,b2,b1,b0])

def sy4_coeffs(h0, h1, h2, h3):
    def T(h3, h2, h1, h0):
        return 2*h0*h3*(bt[1]*(h0-2*h1-h2-h3) + bt[2]*(h0+h1-h2-h3) + bt[3]*(h0+h1+2*h2-h3))

    def C(h3, h2, h1, h0):
        return 0.5*T(h3,h2,h1,h0) + SY4.a[1]*3*h0*np.sqrt(h1*h2)*h3

    bt = SY4.b
    a1 = C(h3,h2,h1,h0)/(h0*h1*(h1+h2+h3))
    a3 = C(h0,h1,h2,h3)/(h3*h2*(h0+h1+h2))

    a2 = - (2*h0*h3*(bt[1]+bt[2]+bt[3]) + a1*h0*(h1+h2+h3) + a3*(h0+h1+h2)*h3)
    a2 /= (h0+h1)*(h2+h3)

    a4 = - (a1*h0 + a2*(h0+h1) + a3*(h0+h1+h2))
    a4 /= h0+h1+h2+h3

    a0 = - (a1+a2+a3+a4)

    return np.array([a0,a1,a2,a3,a4]), h0/h3*SY4.b[:]

class SYAM4(SYAMBase):
    key = "syam4"
    steps = 4
    sy = SY4
    am = AM4

class SYAM6(SYAMBase):
    key = "syam6"
    steps = 6
    sy = SY6
    am = AM6

class SYAM8(SYAMBase):
    key = "syam8"
    steps = 8
    sy = SY8
    am = AM8