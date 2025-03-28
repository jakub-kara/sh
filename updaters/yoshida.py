import numpy as np
from .nuclear import NuclearUpdater
from classes.molecule import Molecule
from dynamics.base import Dynamics
from electronic.base import ESTProgram

class Yoshida4(NuclearUpdater):
    key = "y4"
    substeps = 4
    w0 = -np.cbrt(2) / (2 - np.cbrt(2))
    w1 = 1 / (2 - np.cbrt(2))
    c = np.array([w1/2, (w0+w1)/2, (w0+w1)/2, w1/2])
    d = np.array([w1, w0, w1, 0])

    def update(self, mols: list[Molecule], dt: float):
        dyn = Dynamics()
        est = ESTProgram()
        mol = mols[-1]
        out = self.out

        for i in range(4):
            out.inter[i] = mol.copy_all()
            if i == 0:
                out.inter[i].pos_ad = out.inp.pos_ad + dt * self.c[i] * out.inp.vel_ad
            else:
                out.inter[i].pos_ad = out.inter[i-1].pos_ad + dt * self.c[i] * out.inter[i-1].vel_ad
            est.request(dyn.mode(out))
            est.run(out.inter[i])
            est.read(out.inter[i], ref = mol)
            est.reset_calc()

            dyn.update_quantum(mols + [out.inter[i]], dt)
            dyn.calculate_acceleration(out.inter[i])
            if i == 0:
                out.inter[i].vel_ad = out.inp.vel_ad + dt * self.d[i] * out.inter[i].acc_ad
            else:
                out.inter[i].vel_ad = out.inter[i-1].vel_ad + dt * self.d[i] * out.inter[i].acc_ad