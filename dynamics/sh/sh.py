import numpy as np
from .checker import HoppingUpdater
from dynamics.base import Dynamics
from dynamics.decoherence import Decoherence
from classes.config import Config, get_cfg
from classes.format import fmt
from classes.meta import Singleton
from classes.molecule import Molecule
from classes.trajectory import Trajectory
from classes.timestep import Timestep
from electronic.base import ESTMode, ESTRequest
from updaters.nuclear import CompositeIntegrator
from updaters.coeff import CoeffUpdater
from updaters.tdc import TDCUpdater

@Config.Option("vel", "rescale")
@Config("rescale", default="vel", path=Config.DYNAMICS, valid=Config.STR)
class Rescaler(metaclass = Singleton):
    @staticmethod
    def _normalise(a):
        return a / np.linalg.norm(a)

    def get_delta(self, mol: Molecule):
        return self._normalise(mol.vel_ad)

@ESTRequest(ESTMode(ESTMode.TARGET), this=False)
@Config.Option("nac", "rescale")
class NACRescaler(Rescaler):
    def get_delta(self, mol: Molecule):
        if np.any(np.isnan(mol.nacdr_ssad[mol.active, mol.target])):
            Dynamics().run_est(mol, ref = mol)

        # check this works
        delta = mol.nacdr_ssad[mol.active, mol.target]
        delta /= mol.mass_a[:,None]
        return self._normalise(delta)

@Config.Option("eff", "rescale")
class EffectiveNACRescaler(Rescaler):
    def get_delta(self, mol):
        return self._normalise(mol.eff_nac())

@Config("decoherence", default="none", path=Config.DYNAMICS, valid=Config.STR)
@Config("rescale", default="vel", path=Config.DYNAMICS, valid=Config.STR)
@Config("reverse", default=False, path=Config.DYNAMICS, valid=Config.BOOL)
@Config.Init(HoppingUpdater, Decoherence, Rescaler)
class SurfaceHopping(Dynamics):
    mode = ESTMode(ESTMode.ACTIVE)

    def __init__(self, **config):
        super().__init__(**config)

        self._reverse = get_cfg(self, "reverse", config)

        get_cfg(HoppingUpdater, "prob", config)(**config)
        get_cfg(Decoherence, "decoherence", config)(**config)
        get_cfg(Rescaler, "rescale", config)()

    def step_mode(self, mol):
        return self.mode(mol) + TDCUpdater().mode(mol) + CoeffUpdater().mode(mol)

    def calculate_acceleration(self, mol: Molecule):
        mol.acc_ad = -mol.grad_sad[mol.active] / mol.mass_a[:,None]

    def potential_energy(self, mol: Molecule):
        return mol.ham_eig_ss[mol.active, mol.active]

    def _get_delta(self, mol: Molecule):
        return Rescaler().get_delta(mol)

    def _avail_kinetic_energy(self, mol: Molecule, delta: np.ndarray):
        a, b, _ = self._get_ABC(mol, delta)
        return b**2/(4*a)

    def _has_energy(self, mol: Molecule, delta: np.ndarray):
        return self._avail_kinetic_energy(mol, delta) + mol.ham_eig_ss[mol.active, mol.active] - mol.ham_eig_ss[mol.target, mol.target] > 0

    def _get_ABC(self, mol : Molecule, delta: np.ndarray):
        #to use later
        ediff =  mol.ham_eig_ss[mol.target, mol.target] - mol.ham_eig_ss[mol.active, mol.active]
        a = 0.5 * np.einsum('a,ai->',mol.mass_a, delta**2)
        b = -np.einsum('a,ai,ai->',mol.mass_a, mol.vel_ad, delta)
        c = ediff
        return a, b, c

    def _adjust_velocity(self, mol: Molecule, delta: np.ndarray):
        ediff =  mol.ham_eig_ss[mol.target, mol.target] - mol.ham_eig_ss[mol.active, mol.active]

        # compute coefficients in the quadratic equation
        a = 0.5 * np.einsum('a,ai->',mol.mass_a, delta**2)
        b = -np.einsum('a,ai,ai->',mol.mass_a, mol.vel_ad, delta)
        c = ediff

        # find the determinant
        D = b**2 - 4 * a * c
        if D < 0:
            print('Issue with rescaling...')
            raise RuntimeError
            # if self._reverse:
            #     gamma = -b/a
            # else:
            #     gamma = 0
        # choose the smaller solution of the two
        elif b < 0:
            gamma = -(b + np.sqrt(D)) / (2 * a)
        elif b >= 0:
            gamma = -(b - np.sqrt(D)) / (2 * a)

        mol.vel_ad -= gamma * delta

    def _reverse_velocity(self, mol: Molecule, delta: np.ndarray):
        ediff =  mol.ham_eig_ss[mol.target, mol.target] - mol.ham_eig_ss[mol.active, mol.active]

        # compute coefficients in the quadratic equation
        a = 0.5 * np.einsum('a,ai->',mol.mass_a, delta**2)
        b = -np.einsum('a,ai,ai->',mol.mass_a, mol.vel_ad, delta)
        c = ediff

        # find the determinant
        D = b**2 - 4 * a * c
        if D < 0:
            # reverse if no real solution and flag set
            gamma = -b/a
        else:
            print(self._avail_kinetic_energy(mol,delta))
            print(mol.vel_ad * delta, np.sum((mol.vel_ad*delta)**2)/2,np.sum(mol.vel_ad/np.linalg.norm(mol.vel_ad)*delta/np.linalg.norm(delta)))
            print(mol.vel_ad,'\n',delta,'\n',a,b,c,D)
            raise RuntimeError("Issue with rescaling...")

        mol.vel_ad -= gamma * delta
        CompositeIntegrator().to_init()

    def update_target(self, mols: list[Molecule], ts: Timestep):
        hop = HoppingUpdater()
        hop.run(mols, ts)
        mols[-1].target = hop.hop.out

    # TODO: move to molecule
    def dat_header(self, traj: Trajectory):
        dic = super().dat_header(traj)
        dic["act"] = fmt("Active State")
        return dic

    def dat_dict(self, traj: Trajectory):
        dic = super().dat_dict(traj)
        dic["act"] = fmt(traj.mol.active)
        return dic

    def h5_dict(self, traj: Trajectory):
        dic = super().h5_dict(traj)
        dic["act"] = traj.mol.active
        return dic
