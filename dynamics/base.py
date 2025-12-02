import numpy as np
from abc import abstractmethod
import os, time, json

from classes.bundle import Bundle
from classes.config import Config, get_cfg
from classes.format import *
from classes.meta import Singleton, DecoratorDistributor, Factory
from classes.molecule import Molecule, MoleculeMixin
from classes.out import FileManager as fm, Timer
from classes.timestep import Timestep
from classes.trajectory import Trajectory
from updaters.nuclear import CompositeIntegrator
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater
from electronic.base import ESTProgram, ESTMode, HamTransform, ESTRequest

@Config("method", path=Config.DYNAMICS, valid=Config.STR)
@Config("backstep", default=False, path=Config.DYNAMICS, valid=Config.BOOL)
@Config.Init(Config.Init.THIS, TDCUpdater, CoeffUpdater, CompositeIntegrator, ESTProgram, fm, Timestep)
class Dynamics(DecoratorDistributor, metaclass = Singleton):
    mode = ESTMode()
    _molmixins = []

    def __init__(self, **config):
        self._success = True
        self._backstep = get_cfg(self, "backstep", config)

    # === Dynamics methods ===

    @abstractmethod
    def calculate_acceleration(self, mol: Molecule):
        pass

    @abstractmethod
    def potential_energy(self, mol: Molecule):
        pass

    def total_energy(self, mol: Molecule):
        return self.potential_energy(mol) + mol.kinetic_energy()

    def population(self, mol: Molecule, s: int):
        return np.abs(mol.coeff_s[s])**2

    @Timer(id = "nuc",
        head = "Nuclear Adjustment")
    @abstractmethod
    def adjust_nuclear(self, traj: Trajectory):
        pass

    @abstractmethod
    def step_mode(self, mol: Molecule):
        pass

    def remake_molecule(self):
        Factory.update_methods(Molecule, **self._mol_methods())

    def validate(self, obj):
        self._success &= obj.success

    # === Bundle methods ===
    def prepare_bundle(self, bundle: Bundle):
        config = bundle.config
        self.set_components(**config)

        # TODO: rework
        traj = Trajectory(**config)
        mol = self.create_molecule(mixins = self._molmixins, **config)
        traj.add_molecule(mol)
        traj.set_molecules(**config)
        traj.set_timestep(**config)
        bundle.add_trajectory(traj)
        bundle.save_setup()

        self._write_event("INIT 0", "w")
        self.prepare_trajs(bundle)

    def prepare_trajs(self, bundle: Bundle):
        for itraj, traj in enumerate(bundle.trajs):
            os.chdir(f"{itraj}")
            fm.init_all()
            fm.log(BORDER)
            fm.log(json.dumps(bundle.config, indent=4))
            fm.log()
            self.prepare_traj(traj)
            os.chdir("..")

    def step_bundle(self, bundle: Bundle):
        self._check_stop(bundle)
        self._check_pause(bundle)

        bundle.set_active()
        os.chdir(f"{bundle.iactive}")
        self.step_traj(bundle.active)
        os.chdir("..")

        if self._success:
            self.step_success(bundle)

    def _check_stop(self, bundle: Bundle):
        if os.path.exists("STOP"):
            for itraj, traj in enumerate(bundle.trajs):
                os.chdir(f"{itraj}")
                fm.log("STOP directive detected. Terminating.")
                os.chdir("..")
            self._write_event("STOP")
            exit()

    def _check_pause(self, bundle: Bundle):
        if os.path.exists("PAUSE"):
            self._write_event("PAUSE", newline=False)

            count = 0
            while not os.path.exists("RESUME"):
                time.sleep(60)
                count += 1
            self._write_event(count*60)
            self._write_event("RESUME")

            os.system("rm PAUSE RESUME")

    def _write_event(self, msg, mode = "a", newline = True):
        with open("events.log", mode) as f:
            f.write(str(msg) + "\n"*newline)

    def step_success(self, bundle: Bundle):
        os.chdir(f"{bundle.iactive}")
        bundle.active.next_step()
        bundle.active.save_step()
        bundle.active.write_outputs()
        os.chdir("..")

    # === Trajectory methods ===

    @Timer(
        id = "init",
        head = f"{BORDER}\nTrajectory Initialisation",
        msg = "Total time",
        out = fm.log)
    @ESTRequest()
    def prepare_traj(self, traj: Trajectory):
        mol = traj.mols[-1]
        traj.write_headers()

        self.run_est(mol, mol)
        traj.ref_en = self.total_energy(mol)
        self.calculate_acceleration(mol)

        if self._backstep:
            self.run_backstep(traj.mols, traj.timestep)

        traj.timestep.step_success(traj.mols)
        traj.save_step()
        traj.write_outputs()


    @ESTRequest(TDCUpdater, CoeffUpdater)
    def run_backstep(self, mols: list[Molecule], ts: Timestep):
        ts_back = get_cfg(Timestep, "timestep", {"timestep": "const"})(dt = -ts.dt)
        nupd = CompositeIntegrator()
        mol = mols[-1]

        nupd.run([mol], ts_back)
        for i in range(len(mols) - 1):
            mols[i] = nupd.result.out

        self.run_est(mol, mols[-2])
        self.update_tdc(mols, ts)

        for i in range(len(mols) - 1):
            mols[i] = mol.copy()

    @Timer(
        id = "tot",
        msg = "Total time",
        foot = BORDER,
        out = fm.log)
    def step_traj(self, traj: Trajectory):
        traj.step_header()

        self.update_traj(traj)

        if not self._success:
            return

        self.adjust_nuclear(traj)

    @Timer(
        id = "nuc",
        head = "Nuclear",
        out = fm.log)
    @ESTRequest(CoeffUpdater, TDCUpdater)
    def update_traj(self, traj: Trajectory):
        nupd = CompositeIntegrator()
        self._success = True
        ts = traj.timestep

        nupd.run(traj.mols, ts)
        self.validate(nupd)
        if not self._success:
            os.chdir("est")
            ESTProgram().recover_wf("../backup")
            os.chdir("..")
            return

        temp = nupd.result.out
        ts.validate(traj.mols + [temp])
        self.validate(ts)
        if not self._success:
            ts.step_fail(traj.mols)
            os.chdir("est")
            ESTProgram().recover_wf("../backup")
            os.chdir("..")
            return

        traj.add_molecule(temp)
        traj.pop_molecule(0)
        traj.report_energy()

    # === Molecule methods ===

    @Timer(
        id = "est",
        head = "Electronic Calculation",
        out = fm.log)
    def run_est(self, mol: Molecule, ref = None):
        est = ESTProgram()
        est.run(mol)
        est.read(mol, ref)
        est.reset_calc()

    @Timer(
        id = "qua",
        head = "Quantum Propagation",
        out = fm.log)
    def update_quantum(self, mols: list[Molecule], ts: Timestep):
        self.update_tdc(mols, ts)
        self.update_coeff(mols, ts)

    def update_tdc(self, mols: list[Molecule], ts: Timestep):
        tdcupd = TDCUpdater()
        tdcupd.run(mols, ts)
        mols[-1].nacdt_ss = tdcupd.result.out
        # mols[-1].adjust_tdc(mols[-2])

    def update_coeff(self, mols: list[Molecule], ts: Timestep):
        cupd = CoeffUpdater()
        cupd.run(mols, ts)
        mols[-1].coeff_s = cupd.result.out

    # === Setup ===

    def set_components(self, **config):
        self.set_io(**config)
        self.set_est(**config)
        self.set_nuclear(**config)
        self.set_tdc_updater(**config)
        self.set_coeff_updater(**config)

    @staticmethod
    def set_dynamics(**config):
        Dynamics.reset()
        get_cfg(Dynamics, "method", config)(**config)

    def set_est(self, **config):
        get_cfg(HamTransform, "transform", config)()
        ESTProgram.reset()
        get_cfg(ESTProgram, "program", config)(**config)

    def set_nuclear(self, **config):
        CompositeIntegrator.reset()
        CompositeIntegrator(**config)

    def set_tdc_updater(self, **config):
        TDCUpdater.reset()
        get_cfg(TDCUpdater, "tdc_upd", config)(**config)

    def set_coeff_updater(self, **config):
        CoeffUpdater.reset()
        get_cfg(CoeffUpdater, "coeff_upd", config)(**config)

    def set_io(self, **config):
        fm(**config)

    def create_molecule(self, mixins, **config):
        fac = Factory(Molecule, MoleculeMixin)
        fac.add_mixins(*mixins)
        fac.add_methods(**self._mol_methods())

        mol = fac.create()(n_states=ESTProgram().n_states, **config)
        mol.set_coeff()
        return mol

    def _mol_methods(self):
        return {
            "potential_energy": lambda x: self.potential_energy(x),
            "total_energy": lambda x: self.total_energy(x),
            "population": lambda x, s: self.population(x, s)
        }