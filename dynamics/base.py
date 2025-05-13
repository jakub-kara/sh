import numpy as np
from abc import abstractmethod
from copy import deepcopy
import os, json

from classes.bundle import Bundle
from classes.constants import convert
from classes.meta import Singleton, Selector, DecoratorDistributor, Factory
from classes.molecule import Molecule, MoleculeMixin
from classes.out import Output as out, Timer, Logger, DirChange
from classes.timestep import Timestep
from classes.trajectory import Trajectory
from updaters.composite import CompositeIntegrator
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater
from electronic.base import ESTProgram, ESTMode, HamTransform

class Dynamics(Selector, DecoratorDistributor, metaclass = Singleton):
    mode = ESTMode()

    def __init__(self, **config):
        self._config = config
        self._success = True

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

    # === Bundle methods ===

    def prepare_bundle(self, bundle: Bundle):
        config = bundle.config
        config["nuclear"].setdefault("mixins", [])
        self.set_components(**config)

        # TODO: rework
        traj = Trajectory(**config)
        mol = self.create_molecule(coeff = config["quantum"].get("input", None), **config["nuclear"], **config["dynamics"])
        traj.add_molecule(mol)
        traj.set_molecules(**config["nuclear"])
        traj.set_timestep(**config["dynamics"])
        bundle.add_trajectory(traj)
        bundle.save_setup()

        with open("events.log", "w") as f:
            f.write(f"INIT 0\n")
        self.prepare_trajs(bundle)

    def prepare_trajs(self, bundle: Bundle):
        for itraj, traj in enumerate(bundle.trajs):
            os.chdir(f"{itraj}")
            out.open_log()
            out.write_log(json.dumps(bundle.config, indent=4), "w")
            out.write_log()
            self.prepare_traj(traj)
            out.close_log()
            os.chdir("..")

    def step_bundle(self, bundle: Bundle):
        bundle.set_active()
        print()
        if bundle.n_traj > 1:
            print(bundle.iactive, bundle.n_traj)

        os.chdir(f"{bundle.iactive}")
        out.open_log()
        self.step_traj(bundle.active)
        # bundle.active.write_outputs()
        out.close_log()
        os.chdir("..")

    # === Trajectory methods ===

    @Timer(
        id = "init",
        head = f"{out.border}\nTrajectory Initialisation",
        msg = "Total time",
        out = out.write_log)
    def prepare_traj(self, traj: Trajectory):
        mol = traj.mols[-1]
        traj.write_headers()

        self.run_est(mol, mol, mode = self.step_mode(mol))
        traj.ref_en = self.total_energy(mol)
        self.calculate_acceleration(mol)

        self.update_tdc(traj.mols, traj.timestep)

        for i in range(traj.n_steps - 1):
            traj.mols[i] = traj.mol

        traj.write_outputs()

    @Timer(
        id = "tot",
        msg = "Total time",
        foot = out.border,
        out = out.write_log)
    def step_traj(self, traj: Trajectory):
        traj.step_header()
        traj.save_step()

        self.update_traj(traj)

        if not traj.timestep.success:
            return

        self.adjust_nuclear(traj)
        traj.next_step()

    @Timer(
        id = "nuc",
        head = "Nuclear",
        out = out.write_log)
    def update_traj(self, traj: Trajectory):
        nupd = CompositeIntegrator()
        ts = traj.timestep
        nupd.run(traj.mols, ts)
        temp = nupd.active.out.out

        traj.report_energy()

        ts.validate(traj.mols + [temp])
        if not ts.success:
            ts.step_fail()
            ESTProgram().recover_wf()
            return

        traj.add_molecule(temp)
        traj.pop_molecule(0)

    # === Molecule methods ===

    @Timer(
        id = "est",
        head = "Electronic Calculation",
        out = out.write_log)
    def run_est(self, mol: Molecule, ref = None, mode = ""):
        est = ESTProgram()
        est.request(*mode)
        est.run(mol)
        est.read(mol, ref)
        est.reset_calc()

    @Timer(
        id = "qua",
        head = "Quantum Propagation",
        out = out.write_log)
    def update_quantum(self, mols: list[Molecule], ts: Timestep):
        self.update_tdc(mols, ts)
        self.update_coeff(mols, ts)

    def update_tdc(self, mols: list[Molecule], ts: Timestep):
        tdcupd = TDCUpdater()
        tdcupd.run(mols, ts)
        mols[-1].nacdt_ss = tdcupd.tdc.out

    def update_coeff(self, mols: list[Molecule], ts: Timestep):
        cupd = CoeffUpdater()
        cupd.run(mols, ts)
        mols[-1].coeff_s = cupd.coeff.out

    # === Setup ===

    def set_components(self, *, electronic, nuclear, quantum, output, **kwargs):
        self.set_est(**electronic)
        self.set_nuclear(**nuclear)
        self.set_tdc_updater(**quantum)
        self.set_coeff_updater(**quantum)
        self.set_io(**output)

    @staticmethod
    def set_dynamics(dynamics, **config):
        Dynamics.reset()
        Dynamics.select(dynamics["method"])(dynamics=dynamics, **config)

    def set_est(self, **electronic):
        HamTransform.select(electronic.get("transform", "none"))()
        ESTProgram.reset()
        ESTProgram.select(electronic["program"])(**electronic)

    def set_nuclear(self, **nuclear):
        CompositeIntegrator.reset()
        CompositeIntegrator(nuc_upd = nuclear.get("nuc_upd", "vv"))

    def set_tdc_updater(self, **quantum):
        TDCUpdater.reset()
        TDCUpdater.select(quantum.get("tdc_upd", "npi"))(**quantum)

    def set_coeff_updater(self, **quantum):
        CoeffUpdater.reset()
        CoeffUpdater.select(quantum.get("coeff_upd", "tdc"))(**quantum)

    def set_io(self, **output):
        out.setup(**output)

    def create_molecule(self, mixins, coeff = None, **config):
        fac = Factory(Molecule, MoleculeMixin)
        fac.add_mixins(mixins)
        fac.add_methods(**self._mol_methods())

        mol = fac.create()(n_states=ESTProgram().n_states, **config)
        mol.get_coeff(coeff)
        return mol

    def _mol_methods(self):
        return {
            "potential_energy": lambda x: self.potential_energy(x),
            "total_energy": lambda x: self.total_energy(x),
            "population": lambda x, s: self.population(x, s)
        }