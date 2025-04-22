import numpy as np
from typing import Callable
from .base import Updater
from classes.molecule import Molecule
from classes.timestep import Timestep

class AMBase(Updater):
    substeps = 1
    b = np.empty(1)
    c = None

    def update(self, mols: list[Molecule], ts: Timestep):
        temp = mols[-1].copy_all()
        temp.vel_ad = mols[-1].pos_ad + ts.dt*np.einsum("j,j...->...", self.b, np.array([mol.vel_ad for mol in mols]))
        return temp

class AM2(AMBase):
    name = "am2"
    steps = 2
    order = 2
    b = np.array([1/2, 1/2])
    c = -1/12

class AM3(AMBase):
    name = "am3"
    steps = 3
    order = 3
    b = np.array([-1/12, 2/3, 5/12])
    c = 1/24

class AM4(AMBase):
    name = "am4"
    steps = 4
    order = 4
    b = np.array([1/24, -5/24, 19/24, 3/8])
    c = -19/720

class AM5(AMBase):
    name = "am5"
    steps = 5
    order = 5
    b = np.array([-19/720, 53/360, -11/30, 323/360, 251/720])
    c = 3/160

class AM6(AMBase):
    name = "am6"
    steps = 6
    order = 6
    b = np.array([3/160, -173/1440, 241/720, -133/240, 1427/1440, 95/288])
    c = -863/60480

class AM7(AMBase):
    name = "am7"
    steps = 7
    order = 7
    b = np.array([-863/60480, 263/2520, -6737/20160, 586/945, -15487/20160, 2713/2520, 19087/60480])
    c = 275/24192

class AM8(AMBase):
    name = "am8"
    steps = 8
    order = 8
    b = np.array([275/24192, -11351/120960, 1537/4480, -88547/120960, 123133/120960, -4511/4480, 139849/120960, 5257/17280])
    c = -33953/3628800