import numpy as np
from integrators.integrators import Integrator1
from classes.molecule import Molecule

class AMBase(Integrator1):
    def __init__(self):
        super().__init__()
        self.substeps = 1
        self.b = np.empty(1)
        self.c = 0

    def integrate(self, fun, dt: float, *args: Molecule):
        temp = args[-1].copy_all()
        temp.pos_ad = args[-1].pos_ad + dt*np.einsum("j,j...->...", self.b, np.array([mol.vel_ad for mol in args]))
        return temp

class AM2(AMBase):
    def __init__(self):
        super().__init__()
        self.name = "am2"
        self.steps = 2
        self.order = 2
        self.b = np.array([1/2, 1/2])
        self.c = -1/12

class AM3(AMBase):
    def __init__(self):
        super().__init__()
        self.name = "am3"
        self.steps = 3
        self.order = 3
        self.b = np.array([-1/12, 2/3, 5/12])
        self.c = 1/24

class AM4(AMBase):
    def __init__(self):
        super().__init__()
        self.name = "am4"
        self.steps = 4
        self.order = 4
        self.b = np.array([1/24, -5/24, 19/24, 3/8])
        self.c = -19/720

class AM5(AMBase):
    def __init__(self):
        super().__init__()
        self.name = "am5"
        self.steps = 5
        self.order = 5
        self.b = np.array([-19/720, 53/360, -11/30, 323/360, 251/720])
        self.c = 3/160

class AM6(AMBase):
    def __init__(self):
        super().__init__()
        self.name = "am6"
        self.steps = 6
        self.order = 6
        self.b = np.array([3/160, -173/1440, 241/720, -133/240, 1427/1440, 95/288])
        self.c = -863/60480

class AM7(AMBase):
    def __init__(self):
        super().__init__()
        self.name = "am7"
        self.steps = 7
        self.order = 7
        self.b = np.array([-863/60480, 263/2520, -6737/20160, 586/945, -15487/20160, 2713/2520, 19087/60480])
        self.c = 275/24192

class AM8(AMBase):
    def __init__(self):
        super().__init__()
        self.name = "am8"
        self.steps = 8
        self.order = 8
        self.b = np.array([275/24192, -11351/120960, 1537/4480, -88547/120960, 123133/120960, -4511/4480, 139849/120960, 5257/17280])
        self.c = -33953/3628800