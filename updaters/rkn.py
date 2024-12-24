import numpy as np
from .nuclear import NuclearUpdater
from .updaters import UpdateResult
from classes.molecule import Molecule
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram

class RKNBase(NuclearUpdater):
    a = np.empty((1,1))
    b = np.empty(1)
    c = np.empty(1)
    d = np.empty(1)

    def update(self, mols: list[Molecule], dt: float, dyn: Dynamics):
        # helper function for triangular numbers
        def tri(x):
            return x * (x + 1) // 2

        mol = mols[-1]
        out = UpdateResult(mol, self.substeps)

        out.inter[0] = mol
        # RKN integration substep-by-substep
        for i in range(1, self.substeps):
            print(f"Substep {i}")
            # evaluate intermediate position and acceleration
            out.inter[i] = mol.copy_all()
            out.inter[i].pos_ad += dt * self.c[i] * mol.vel_ad + dt**2 * np.einsum("j,j...->...", self.a[tri(i-1):tri(i)], [m.acc_ad for m in out.inter[:i]])

            est = ESTProgram()
            dyn.setup_est(mode = dyn.get_mode())
            est.run(out.inter[i])
            est.read(out.inter[i], ref = mol)
            est.reset_calc()

            dyn.update_quantum(mols + [out.inter[i]], dt * self.c[i])
            dyn.calculate_acceleration(out.inter[i])

        print("Final Step")
        # find new position and velocity from all the substeps
        temp = mol.copy_all()
        temp.pos_ad += dt * mol.vel_ad + dt**2 * np.einsum("j,j...->...", self.b, [m.acc_ad for m in out.inter])
        temp.vel_ad += dt * np.einsum("j,j...->...", self.d, [m.acc_ad for m in out.inter])

        # calculate new acceleration
        est = ESTProgram()
        dyn.setup_est(mode = dyn.get_mode())
        est.run(temp)
        est.read(temp, ref = mol)
        est.reset_calc()

        dyn.update_quantum(mols + [temp], dt)
        dyn.calculate_acceleration(temp)

        return temp

class RKN4(RKNBase, key = "rkn4"):
    substeps = 4
    a = np.array([
        1/18,
        0,
        2/9,
        1/3,
        0,
        1/6
    ])
    c = np.array([0,    1/3,    2/3,    1])
    b = np.array([13/120,   3/10,   3/40,   1/60])
    d = np.array([1/8,      3/8,    3/8,    1/8])

class RKN4b(RKNBase, key = "rkn4b"):
    substeps = 3
    a = np.array([
        1/8,
        0,
        1/2
    ])
    c = np.array([0, 1/2, 1])
    b = np.array([1/6, 1/3, 0])
    d = np.array([1/6, 2/3, 1/6])

class RKN6(RKNBase, key = "rkn6"):
    substeps = 7
    order = 6
    a = np.array([
        1/200,
        1/150,
        1/75,
        2/75,
        0,
        4/75,
        9/200,
        0,
        9/100,
        9/200,
        199/3600,
        -19/150,
        47/120,
        -119/1200,
        89/900,
        -179/1824,
        17/38,
        0,
        -37/152,
        219/456,
        -157/1824
    ])
    c = np.array([0,    1/10,   1/5,    2/5,    3/5,    4/5,    1])
    b = np.array([61/1008,  0,  475/2016,   25/504, 125/1008, 25/1008, 11/2016])
    d = np.array([19/288,   0,  25/96,      25/144, 25/144,   25/96,   19/288])

class RKN8(RKNBase, key = "rkn8"):
    name = "rkn8"
    substeps = 11
    a = np.array([
        49/12800,

        49/9600,
        49/4800,

        16825/381024,
        -625/11907,
        18125/190512,

        23/840,
        0,
        50/609,
        9/580,

        533/68040,
        0,
        5050/641277,
        -19/5220,
        23/12636,

        -4469/85050,
        0,
        -2384000/641277,
        3896/19575,
        -1451/15795,
        502/135,

        694/10125,
        0,
        0,
        -5504/10125,
        424/2025,
        -104/2025,
        364/675,

        30203/691200,
        0,
        0,
        0,
        9797/172800,
        79391/518400,
        20609/345600,
        70609/2073600,

        1040381917/14863564800,
        0,
        548042275/109444608,
        242737/5345280,
        569927617/6900940800,
        -2559686731/530841600,
        -127250389/353894400,
        -53056229/2123366400,
        23/5120,

        -33213637/179088000,
        0,
        604400/324597,
        63826/445875,
        0,
        -6399863/2558400,
        110723/511680,
        559511/35817600,
        372449/7675200,
        756604/839475
    ])
    c = np.array([0, 7/80, 7/40, 5/12, 1/2, 1/6, 1/3, 2/3, 5/6, 1/12, 1])
    b = np.array([121/4200, 0, 0, 0, 43/525, 33/350, 17/140, 3/56, 31/1050, 512/5775, 1/550])
    d = np.array([41/840, 0, 0, 0, 34/105, 9/35, 9/280, 9/280, 9/35, 0, 41/840])