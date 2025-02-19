import numpy as np
from .nuclear import NuclearUpdater
from classes.molecule import Molecule
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram

class Yoshida4(NuclearUpdater):
    key = "y4"
    substeps = 4
    w0 = -np.cbrt(2) / (2 - np.cbrt(2))
    w1 = 1 / (2 - np.cbrt(2))
    c = np.array([w1/2, (w0+w1)/2, (w0+w1)/2, w1/2])
    d = np.array([w1, w0, w1, 0])

    def update(self, mols: list[Molecule], dt: float, dyn: Dynamics):
        mol = mols[-1]
        out = self.out
        est = ESTProgram()

        for i in range(4):
            out.inter[i] = mol.copy_all()
            if i == 0:
                out.inter[i].pos_ad = out.inp.pos_ad + dt * self.c[i] * out.inp.vel_ad
            else:
                out.inter[i].pos_ad = out.inter[i-1].pos_ad + dt * self.c[i] * out.inter[i-1].vel_ad
            dyn.setup_est(mode = dyn.get_mode())
            est.run(out.inter[i])
            est.read(out.inter[i], ref = mol)
            est.reset_calc()

            dyn.update_quantum(mols + [out.inter[i]], dt)
            dyn.calculate_acceleration(out.inter[i])
            if i == 0:
                out.inter[i].vel_ad = out.inp.vel_ad + dt * self.d[i] * out.inter[i].acc_ad
            else:
                out.inter[i].vel_ad = out.inter[i-1].vel_ad + dt * self.d[i] * out.inter[i].acc_ad

        # # x1, a1, v1
        # out.inter[0] = mol.copy_all()
        # out.inter[0].pos_ad += self.c[0] * out.inter[0].vel_ad * dt
        # dyn.setup_est(mode = dyn.get_mode())
        # est.run(out.inter[0])
        # est.read(out.inter[0], ref = mol)
        # est.reset_calc()

        # dyn.update_quantum(mols + [out.inter[0]], dt)
        # dyn.calculate_acceleration(out.inter[0])
        # out.inter[0].vel_ad += self.d[0] * out.inter[0].acc_ad * dt

        # # x2, a2, v2
        # out.inter[1] = out.inter[0].copy_all()
        # out.inter[1].pos_ad += self.c[1] * out.inter[1].vel_ad * dt
        # dyn.setup_est(mode = dyn.get_mode())
        # est.run(out.inter[1])
        # est.read(out.inter[1], ref = mol)
        # est.reset_calc()

        # dyn.update_quantum(mols + [out.inter[1]], dt)
        # dyn.calculate_acceleration(out.inter[1])
        # out.inter[1].vel_ad += self.d[1] * out.inter[1].acc_ad * dt

        # # x3, a3, v3
        # out.inter[2] = out.inter[1].copy_all()
        # out.inter[2].pos_ad += self.c[2] * out.inter[2].vel_ad * dt
        # dyn.setup_est(mode = dyn.get_mode())
        # est.run(out.inter[2])
        # est.read(out.inter[2], ref = mol)
        # est.reset_calc()

        # dyn.update_quantum(mols + [out.inter[2]], dt)
        # dyn.calculate_acceleration(out.inter[2])
        # out.inter[2].vel_ad += self.d[2] * out.inter[2].acc_ad * dt

        # out.inter[3] = out.inter[2].copy_all()
        # out.inter[3].pos_ad += self.c[3] * out.inter[3].vel_ad * dt
        # dyn.setup_est(mode = dyn.get_mode())
        # est.run(out.inter[3])
        # est.read(out.inter[3], ref = mol)
        # est.reset_calc()

        # dyn.update_quantum(mols + [out.inter[3]], dt)
        # dyn.calculate_acceleration(out.inter[3])