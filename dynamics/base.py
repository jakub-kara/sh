import numpy as np

from abc import abstractmethod
from copy import deepcopy
from classes.constants import convert
from classes.meta import SingletonFactory
from classes.molecule import Molecule
from classes.out import Output, Printer, Timer
from classes.trajectory import Trajectory
from updaters.composite import CompositeIntegrator
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater
from electronic.base import ESTProgram, ESTMode

class Dynamics(metaclass = SingletonFactory):
    mode = ESTMode()

    @abstractmethod
    def calculate_acceleration(self, mol: Molecule):
        pass

    @abstractmethod
    def potential_energy(self, mol: Molecule):
        pass

    def total_energy(self, mol: Molecule):
        return self.potential_energy(mol) + mol.kinetic_energy

    def energy_diff(self, mol: Molecule, ref: Molecule):
        return np.abs(self.total_energy(mol) - self.total_energy(ref))

    def population(self, mol: Molecule, s: int):
        return np.abs(mol.coeff_s[s])**2

    @Timer(id = "init",
           head = f"{Output.border}\nTrajectory Initialisation",
           msg = "Total time")
    def prepare_traj(self, traj: Trajectory):
        mol = traj.mols[-1]
        self.write_headers(traj)

        self.steps_elapsed(-1)
        self.run_est(mol, mol, mode = self.mode(mol))
        self.calculate_acceleration(mol)

        self.update_quantum(traj.mols, traj.timestep.dt)

        self.write_outputs(traj)

    @Timer(id = "tot",
           msg = "Total time",
           foot = Output.border)
    def run_step(self, traj: Trajectory):
        self.steps_elapsed(traj.timestep.step)
        traj.step_header()
        traj.save_step()

        self.update_nuclear(traj.mols, traj.timestep.dt)

        nupd = CompositeIntegrator()
        if not nupd.success:
            nupd.to_init()
            ESTProgram().recover_wf()
            print("nupd")
            return

        temp = nupd.active.out.out
        valid = traj.timestep.validate(self.energy_diff(temp, traj.mol))
        if not valid:
            traj.timestep.fail()
            ESTProgram().recover_wf()
            print("timestep")
            return
        traj.add_molecule(temp)
        traj.pop_molecule(0)

        out = Output()
        out.write_log(f"Total energy:   {convert(self.total_energy(traj.mol), 'au', 'ev'):.6f} eV")
        out.write_log(f"Energy shift:   {convert(self.energy_diff(traj.mol, traj.mols[-2]), 'au', 'ev'):.6f} eV")
        out.write_log()

        self.adjust_nuclear(traj.mols, traj.timestep.dt)

        traj.next_step()
        traj.timestep.success()
        traj.timestep.save_nupd()
        self.write_outputs(traj)

        if traj.is_finished:
            traj.save_step()

    @Timer(id = "nuc",
           head = "Nuclear Adjustment")
    @abstractmethod
    def adjust_nuclear(self, mol: Molecule, dt: float):
        pass

    def steps_elapsed(self, steps: int):
        TDCUpdater().elapsed(steps + 1)
        CoeffUpdater().elapsed(steps + 1)

    @abstractmethod
    def step_mode(self, mol: Molecule):
        pass

    @Timer(id = "est",
           head = "Electronic Calculation")
    def run_est(self, mol: Molecule, ref = None, mode = ""):
        est = ESTProgram()
        est.request(*mode)
        est.run(mol)
        est.read(mol, ref)
        est.reset_calc()

    def update_nuclear(self, mols: list[Molecule], dt: float):
        nupd = CompositeIntegrator()
        nupd.run(mols, dt)
        temp = nupd.active.out.out
        nupd.validate(self.energy_diff(temp, mols[-1]))

    @Timer(id = "qua",
           head = "Quantum Propagation")
    def update_quantum(self, mols: list[Molecule], dt: float):
        self.update_tdc(mols, dt)
        self.update_coeff(mols, dt)

    def update_tdc(self, mols: list[Molecule], dt: float):
        tdcupd = TDCUpdater()
        tdcupd.run(mols, dt)
        mols[-1].nacdt_ss = tdcupd.tdc.out

    def update_coeff(self, mols: list[Molecule], dt: float):
        cupd = CoeffUpdater()
        cupd.run(mols, dt)
        mols[-1].coeff_s = cupd.coeff.out

    def _eff_nac(self, mol: Molecule):
        nac_eff = np.zeros_like(mol.nacdr_ssad)
        for i in range(mol.n_states):
            for j in range(i):
                diff = mol.grad_sad[i] - mol.grad_sad[j]
                if np.abs(mol.nacdt_ss[i,j]) < 1e-8:
                    alpha = 0
                else:
                    alpha = (mol.nacdt_ss[i,j] - np.sum(diff * mol.vel_ad)) / np.sum(mol.vel_ad**2)
                nac_eff[i,j] = diff + alpha * mol.vel_ad
                nac_eff[j,i] = -nac_eff[i,j]
        return nac_eff

    def split_mol(self, mol: Molecule):
        out1 = deepcopy(mol)
        out1.coeff_s[:] = 0
        out1.coeff_s[self.split] = mol.coeff_s[self.split]
        out1.coeff_s /= np.sqrt(np.sum(np.abs(out1.coeff_s)**2))

        out2 = deepcopy(mol)
        out2.coeff_s[self.split] = 0
        out2.coeff_s /= np.sqrt(np.sum(np.abs(out2.coeff_s)**2))
        return out1, out2

    def write_headers(self, traj: Trajectory):
        out = Output()
        out.write_dat(self.dat_header(traj), "w")
        out.write_h5(self.h5_info(traj), "w")
        out.write_xyz("", "w")
        out.write_dist("", "w")

    @Timer(id = "out",
           head = "Outputs")
    def write_outputs(self, traj: Trajectory):
        out = Output()
        out.write_dat(self.dat_dict(traj))
        out.write_h5(self.h5_dict(traj))
        out.write_xyz(self.vxyz_string(traj))
        out.write_dist(self.dist_string(traj))

    def dat_header(self, traj: Trajectory):
        nst = traj.mol.n_states

        dic = {}
        dic["time"] = "#" + Printer.write("Time [fs]", "s")

        dic["pop"] = "".join([Printer.write(f"Population {i}", "s") for i in range(nst)])
        dic["pes"] = "".join([Printer.write(f"Pot En {i} [eV]", "s") for i in range(nst)])
        dic["ken"] = Printer.write("Total Kin En [eV]", "s")
        dic["pen"] = Printer.write("Total Pot En [eV]", "s")
        dic["ten"] = Printer.write("Total En [eV]", "s")
        dic["nacdr"] = "".join([Printer.write(f"NACdr {j}-{i} [au]", "s") for i in range(nst) for j in range(i)])
        dic["nacdt"] = "".join([Printer.write(f"NACdt {j}-{i} [au]", "s") for i in range(nst) for j in range(i)])
        dic["coeff"] = "".join([Printer.write(f"Coeff {i}", f" <{Printer.field_length*2+1}") for i in range(nst)])
        return dic

    def dat_dict(self, traj: Trajectory):
        mol = traj.mol
        nst = mol.n_states

        dic = {}
        dic["time"] = Printer.write(convert(traj.timestep.time, "au", "fs"), "f")

        dic["pop"] = "".join([Printer.write(self.population(mol, i), "f") for i in range(nst)])
        dic["pes"] = "".join([Printer.write(convert(mol.ham_eig_ss[i,i], "au", "ev"), "f") for i in range(nst)])
        dic["ken"] = Printer.write(convert(mol.kinetic_energy, "au", "ev"), "f")
        dic["pen"] = Printer.write(convert(self.potential_energy(mol), "au", "ev"), "f")
        dic["ten"] = Printer.write(convert(self.total_energy(mol), "au", "ev"), "f")
        dic["nacdr"] = "".join([Printer.write(mol.nac_norm_ss[i,j], "f") for i in range(nst) for j in range(i)])
        dic["nacdt"] = "".join([Printer.write(mol.nacdt_ss[i,j], "f") for i in range(nst) for j in range(i)])
        dic["coeff"] = "".join([Printer.write(mol.coeff_s[i], "z") for i in range(nst)])
        return dic

    def dist_string(self, traj: Trajectory):
        return traj.mol.to_dist()

    def xyz_string(self, traj: Trajectory):
        return traj.mol.to_xyz()

    def vxyz_string(self, traj: Trajectory):
        return traj.mol.to_vxyz()

    def h5_info(self, traj: Trajectory):
        mol = traj.mol
        dic = {}
        dic["step"] = "info"
        dic["nst"] = mol.n_states
        dic["nat"] = mol.n_atoms
        dic["ats"] = mol.name_a
        dic["mass"] = mol.mass_a
        return dic

    def h5_dict(self, traj: Trajectory):
        mol = traj.mol
        dic = {}
        dic["step"] = traj.timestep.step
        dic["time"] = traj.timestep.time
        dic["pos"] = mol.pos_ad
        dic["vel"] = mol.vel_ad
        dic["acc"] = mol.acc_ad
        dic["trans"] = mol.trans_ss
        dic["hdiag"] = mol.ham_eig_ss
        dic["grad"] = mol.grad_sad
        dic["nacdr"] = mol.nacdr_ssad
        dic["nacdt"] = mol.nacdt_ss
        dic["coeff"] = mol.coeff_s
        return dic
