import numpy as np
from scipy.linalg import expm
from .updaters import Updater, Multistage, UpdateResult
from .tdc import TDCUpdater
from classes.meta import Singleton, SingletonFactory
from classes.molecule import Molecule, BlochMixin

class CoeffUpdater(Updater, metaclass = SingletonFactory):
    mode = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.coeff = None
        self.prop = None

    def new_result(self, mol):
        self.coeff = UpdateResult(mol.coeff_s, self.substeps)
        self.prop = UpdateResult(np.eye(mol.n_states, dtype=np.complex128), self.substeps)

    def no_update(self, mols: list[Molecule], dt: float):
        self.coeff.fill()
        self.prop.fill()

class BlankCoeffUpdater(CoeffUpdater):
    key = "none"

    def update(self, mols, dt):
        self.no_update(mols, dt)

class CoeffTDCUpdater(Multistage, CoeffUpdater):
    key = "tdc"
    steps = 1
    mode = ""

    def update(self, mols: list[Molecule], dt: float):
        tdcupd = TDCUpdater()
        coeff = self.coeff.inp
        prop = self.prop.inp

        for i in range(self.substeps):
            frac = (i + 0.5) / self.substeps
            ham = frac * mols[-1].ham_eig_ss + (1 - frac) * mols[-2].ham_eig_ss
            arg = -(1.j * ham + tdcupd.tdc.interpolate(frac)) * dt / self.substeps
            prop = expm(arg) @ prop
            self.prop.inter[i] = prop
            self.coeff.inter[i] = prop @ coeff

class CoeffLDUpdater(Multistage, CoeffUpdater):
    key = "ld"
    steps = 2
    mode = "o"

    def update(self, mols: list[Molecule], dt: float):
        coeff = self.coeff.inp
        prop = self.prop.inp

        H_tr = mols[-1].ovlp_ss @ mols[-1].ham_eig_ss @ mols[-1].ovlp_ss.T
        for i in range(self.substeps):
            frac = (i + 0.5) / self.substeps
            ham = frac * (H_tr - mols[-2].ham_eig_ss) + mols[-2].ham_eig_ss
            prop = expm(-1j*ham * dt / self.substeps) @ prop
            self.prop.inter[i] = prop
            self.coeff.inter[i] = prop @ coeff

class BlochUpdater(Multistage, Updater, metaclass = Singleton):
    steps = 2
    mode = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bloch = None

    def new_result(self, mol: Molecule, *args, **kwargs):
        self.bloch = UpdateResult(mol.bloch_n3, self.substeps)

    def update(self, mols: list[Molecule], dt: float, active: int):
        tdcupd = TDCUpdater()
        bloch = self.bloch.inp
        nst = mols[-1].n_states

        for i in range(self.substeps):
            frac = (i + 0.5) / self.substeps
            tdc = tdcupd.tdc.interpolate(frac)
            for s in range(nst):
                if s == active:
                    bloch[s, :] = None
                    continue
                ham = frac * mols[-1].ham_eig_ss + (1 - frac) * mols[-2].ham_eig_ss
                mat = np.zeros((3, 3))
                mat[0,1] = ham[s, s] - ham[active, active]
                mat[1,0] = -mat[0,1]
                mat[0,2] = 2 * tdc[active, s]
                mat[2,0] = -mat[0,2]

                bloch[s] = expm(mat * dt / self.substeps) @ bloch[s]
                # print(bloch[s])
                self.bloch.inter[i,s] = bloch[s]

    def no_update(self, mols: list[Molecule], dt: float, active: int):
        self.bloch.fill()