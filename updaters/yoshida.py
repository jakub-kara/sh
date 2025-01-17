import numpy as np
from .nuclear import NuclearUpdater
from classes.molecule import Molecule
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram

class Yoshida4(NuclearUpdater, key = "y4"):
    w0 = -np.cbrt(2) / (2 - np.cbrt(2))
    w1 = 1 / (2 - np.cbrt(2))
    c = np.array([w1/2, (w0+w1)/2, (w0+w1)/2, w1/2])
    d = np.array([w1, w0, w1])

    def update(self, mols: list[Molecule], dt: float, dyn: Dynamics):
        mol = mols[-1]
        out: Molecule = self.out.inp.copy_all()
        est = ESTProgram()

        # x1, a1, v1
        out.pos_ad += self.c[0] * out.vel_ad * dt
        dyn.setup_est(mode = dyn.get_mode())
        est.run(out)
        est.read(out, ref = mol)
        est.reset_calc()

        dyn.update_quantum(mols + [out], dt)
        dyn.calculate_acceleration(out)
        out.vel_ad += self.d[0] * out.acc_ad * dt

        # x2, a2, v2
        out.pos_ad += self.c[1] * out.vel_ad * dt
        dyn.setup_est(mode = dyn.get_mode())
        est.run(out)
        est.read(out, ref = mol)
        est.reset_calc()

        dyn.update_quantum(mols + [out], dt)
        dyn.calculate_acceleration(out)
        out.vel_ad += self.d[1] * out.acc_ad * dt

        # x3, a3, v3
        out.pos_ad += self.c[2] * out.vel_ad * dt
        dyn.setup_est(mode = dyn.get_mode())
        est.run(out)
        est.read(out, ref = mol)
        est.reset_calc()

        dyn.update_quantum(mols + [out], dt)
        dyn.calculate_acceleration(out)
        out.vel_ad += self.d[2] * out.acc_ad * dt

        out.pos_ad += self.c[3] * out.vel_ad * dt
        self.out.out = out