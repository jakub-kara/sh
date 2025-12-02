import numpy as np
from scipy.linalg import expm
from .base import Updater, Multistage, UpdateResult
from .tdc import TDCUpdater
from classes.config import Config, get_cfg
from classes.meta import Singleton
from classes.molecule import Molecule
from classes.timestep import Timestep
from electronic.base import ESTMode

@Config("coeff_upd", default="tdc", path=Config.QUANTUM, valid=Config.STR)
class CoeffUpdater(Updater, metaclass = Singleton):
    mode = ESTMode(ESTMode.NONE)

    def new_result(self, mol):
        self.result = UpdateResult(mol.coeff_s, self.substeps)
        self.prop = UpdateResult(np.eye(mol.n_states, dtype=np.complex128), self.substeps)

    def no_update(self, mols: list[Molecule], ts: Timestep):
        self.result.fill()
        self.prop.fill()

@Config.Option("none", "coeff_upd")
class BlankCoeffUpdater(CoeffUpdater):
    def update(self, mols, ts):
        self.no_update(mols, ts)

@Config.Option("tdc", "coeff_upd")
@Config("prop", default="expm", path=Config.QUANTUM, valid=Config.STR)
class CoeffTDCUpdater(Multistage, CoeffUpdater):
    steps = 1

    def __init__(self, **config):
        super().__init__(**config)
        self.__class__.update = get_cfg(self, "prop", config)

    @Config.Option("rk4", "prop", "CoeffTDCUpdater")
    def _update_rk4(self, mols: list[Molecule], ts: Timestep):
        def linint(a, b, frac):
            return (1-frac) * a + frac * b

        def func(ham, coeff):
            return -1j*ham @ coeff

        raise RuntimeError("This is currently broken.")

        iham = mols[-2].ham_eig_ss - 1j*mols[-2].nacdt_ss
        fham = mols[-1].ham_eig_ss - 1j*mols[-1].nacdt_ss
        ico = mols[-2].coeff_s

        k1 = func(linint(iham, fham, 0), ico)
        k2 = func(linint(iham, fham, 0.5), ico + ts.dt * k1 / 2)
        k3 = func(linint(iham, fham, 0.5), ico + ts.dt * k2 / 2)
        k4 = func(linint(iham, fham, 1), ico + ts.dt * k3)

        out = ico + ts.dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        self.result.out = out / np.linalg.norm(out)

    @Config.Option("expm", "prop", "CoeffTDCUpdater")
    def _update_expm(self, mols: list[Molecule], ts: Timestep):
        tdcupd = TDCUpdater()
        coeff = self.result.inp
        prop = self.prop.inp

        for i in range(self.substeps):
            frac = (i + 0.5) / self.substeps
            ham = frac * mols[-1].ham_eig_ss + (1 - frac) * mols[-2].ham_eig_ss
            arg = -(1.j * ham + tdcupd.result.interpolate(frac)) * ts.dt / self.substeps
            prop = expm(arg) @ prop
            self.prop.push(prop)
            self.result.push(prop @ coeff)

@Config.Option("ld", "coeff_upd")
class CoeffLDUpdater(Multistage, CoeffUpdater):
    steps = 1
    mode = ESTMode(ESTMode.OVERLAP)

    def update(self, mols: list[Molecule], ts: Timestep):
        coeff = self.result.inp
        prop = self.prop.inp

        H_tr = mols[-1].ovlp_ss @ mols[-1].ham_eig_ss @ mols[-1].ovlp_ss.T
        for i in range(self.substeps):
            frac = (i + 0.5) / self.substeps
            ham = frac * (H_tr - mols[-2].ham_eig_ss) + mols[-2].ham_eig_ss
            prop = expm(-1j*ham * ts.dt / self.substeps) @ prop
            self.prop.push(prop)
            self.result.push(prop @ coeff)

@Config.Option("bloch", "coeff_upd")
class BlochUpdater(Multistage, CoeffUpdater):
    steps = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bloch = None

    def new_result(self, mol: Molecule, *args, **kwargs):
        self.result = UpdateResult(mol.bloch_n3, self.substeps)

    def update(self, mols: list[Molecule], ts: Timestep):
        tdcupd = TDCUpdater()
        bloch = self.result.inp
        mol = mols[-1]
        nst = mol.n_states
        act = mol.active

        for k in range(self.substeps):
            frac = (k + 0.5) / self.substeps
            ham = frac * mols[-1].ham_eig_ss + (1 - frac) * mols[-2].ham_eig_ss
            tdc = tdcupd.result.interpolate(frac)

            for s in range(nst):
                if s == act:
                    bloch[s, :] = None
                    continue
                mat = np.zeros((3, 3))
                mat[0,1] = ham[s, s] - ham[act, act]
                mat[1,0] = -mat[0,1]
                mat[0,2] = 2 * tdc[act, s]
                mat[2,0] = -mat[0,2]

                bloch[s] = expm(mat * ts.dt / self.substeps) @ bloch[s]
            self.result.push(bloch[s])

    def no_update(self, mols: list[Molecule], ts: Timestep):
        self.result.fill()