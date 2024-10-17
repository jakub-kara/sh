import numpy as np
import os, pickle
from functools import partial
from abc import ABC, abstractmethod
from classes.molecule import Molecule
from classes.out import Printer, Outputs, record_time
from classes.constants import Constants
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram
from electronic import select_est
from integrators.composite import CompositeIntegrator
from integrators.qtemp import QuantumUpdater
from integrators import select_nucupd

class Trajectory(ABC):
    def __init__(self):
        self._molecules: list[Molecule] = []
        self._nucupd: CompositeIntegrator = None
        self._est: ESTProgram = None
        self._qupd: QuantumUpdater = None
        self._out: Outputs = None

        self.index = None
        self._split = None

    @property
    def n_steps(self):
        return len(self._molecules)

    @property
    def mol(self):
        return self._molecules[-1]

    @property
    def mols(self):
        return self._molecules

    @property
    def n_states(self):
        return self.mol.n_states

    @property
    def split_states(self):
        return self._split

    @property
    def n_atoms(self):
        return self.mol.n_atoms

    def add_molecule(self, mol: Molecule):
        self._molecules.append(mol)
        return self

    def pop_molecule(self, index: int):
        self._molecules.pop(index)
        return self

    def remove_molecule(self, mol: Molecule):
        self._molecules.remove(mol)
        return self

    def split_traj(self):
        pass

    @property
    def pos_nad(self):
        return np.array([mol.pos_ad for mol in self._molecules])

    @pos_nad.setter
    def pos_nad(self, value: np.ndarray):
        for imol, mol in enumerate(self._molecules):
            mol.pos_ad = value[imol]

    @property
    def vel_nad(self):
        return np.array([mol.vel_ad for mol in self._molecules])

    @vel_nad.setter
    def vel_nad(self, value: np.ndarray):
        for imol, mol in enumerate(self._molecules):
            mol.vel_ad = value[imol]

    @property
    def acc_nad(self):
        return np.array([mol.acc_ad for mol in self._molecules])

    @acc_nad.setter
    def acc_nad(self, value: np.ndarray):
        for imol, mol in enumerate(self._molecules):
            mol.acc_ad = value[imol]

    @property
    def coeff_ns(self):
        return np.array([mol.pes.coeff_s for mol in self._molecules])

    @coeff_ns.setter
    def coeff_ns(self, value: np.ndarray):
        for imol, mol in enumerate(self._molecules):
            mol.pes.coeff_s = value[imol]

class TrajectoryDynamics(Dynamics, Trajectory, ABC):
    def __init__(self, *, dynamics: dict, **config):
        super().__init__(**dynamics)
        self.bind_components(dynamics=dynamics, **config)

    def prepare_traj(self):
        self.write_headers()
        self.run_est(self.mol, mode=self._qupd.mode)

    def propagate(self):
        self.update_nuclear()
        self.update_quantum()

        self.pop_molecule(0)
        self.adjust_nuclear()
        # self._ctrl.check_energy(np.abs(self.total_energy(self.mol) - self.total_energy(self.mols[-2])))
        # if self._ctrl.step_ok:
        #     self.pop_molecule(0)
        #     self._nucupd.to_normal()
        #     self.adjust_nuclear()
        # else:
        #     # print("interpolating")
        #     # self.interpolate_wf(self.mols[-2], self.mols[-1], 5)
        #     self._est.recover_wf()
        #     self.pop_molecule(-1)
        #     self._nucupd.state += 1
        #     self.propagate()

    @abstractmethod
    def adjust_nuclear(self):
        pass

    def bind_components(self, *, electronic: dict, nuclear: dict, quantum: dict, output: dict, **config):
        self.bind_est(**electronic)
        self.bind_nuclear_integrator(nuclear["nucupd"])
        self.bind_molecules(**nuclear)
        self.bind_quantum_updater(**quantum)
        self.bind_io(**output)

    def bind_molecules(self, **nuclear):
        mol = Molecule(n_states=self._est.n_states, **nuclear)
        self.add_molecule(mol)

        for _ in range(self._nucupd.steps):
            self.add_molecule(mol.copy_all())

    def bind_est(self, **electronic):
        est: ESTProgram = select_est(electronic["program"])(**electronic)
        self._est = est

    def bind_nuclear_integrator(self, type: str):
        nucupd = select_nucupd(type)
        self._nucupd = nucupd
        self._nucupd.to_init()

    def bind_quantum_updater(self, **quantum):
        qupd = QuantumUpdater(**quantum)
        self._qupd = qupd

    def bind_io(self, **output):
        out = Outputs(**output)
        self._out = out
        self.bind_timers(output["timer"])

    def run_est(self, mol: Molecule, mode: str):
        self.calc_est(mol, mode)
        self.read_est(mol)

    def calc_est(self, mol: Molecule, mode: str):
        os.chdir("est")
        self.setup_est(self._est, mode)
        self._est.write(mol)
        self._est.execute()
        os.chdir("..")

    def read_est(self, mol: Molecule):
        os.chdir("est")
        mol.pes.ham_dia_ss = self._est.read_ham()
        mol.pes.adjust_energy()

        if self._est.any_nacs():
            mol.pes.nacdr_ssad = self._est.read_nac()
            mol.pes.adjust_nacs(self.mols[-2].pes)

        if self._est.any_grads():
            mol.pes.grad_sad = self._est.read_grad()
            self.calculate_acceleration(mol)

        if self._est.any_ovlp():
            mol.pes.ovlp_ss = self._est.read_ovlp(self.mol.name_a.astype("<U2"), self.mols[-1].pos_ad, self.mols[-2].pos_ad)

        mol.pes.transform(False)
        self._est.reset_calc()
        os.chdir("..")

    def interpolate_wf(self, mol_ini: Molecule, mol_fin: Molecule, num: int):
        for i in range(num):
            mol = mol_ini.copy_all()
            mol.pos_ad = (mol_ini.pos_ad * (num - i) + mol_fin.pos_ad * (i + 1))/(num + 1)
            self.run_est(mol, "energy")
        self.run_est(mol_fin)

    def total_energy(self, mol: Molecule):
        return self.potential_energy(mol) + mol.kinetic_energy

    def save_step(self):
        self._est.backup_wf()

        with open("backup/traj.pkl", "wb") as pkl:
            pickle.dump(self, pkl)

    @staticmethod
    def load_step(file):
        with open(file, "rb") as pkl:
            traj: Trajectory = pickle.load(pkl)
        return traj

    def update_nuclear(self):
        temp = self._nucupd.integrate(partial(self.run_est, mode=self._get_mode()), self._dt, *self._molecules)
        self.add_molecule(temp)
        return self

    def _get_mode(self):
        return self._qupd.mode

    def update_quantum(self):
        self._qupd.update(self.mols, self._dt, self._step)
        return self

    def write_headers(self):
        self._out.write_dat(self.dat_header(self._out.record), "w")
        self._out.write_log(self.log_header(), "w")
        self._out.write_mat(self.h5_info(), "w")
        self._out.write_xyz("", "w")

    def write_outputs(self):
        self._out.write_dat(self.dat_dict(self._out.record))
        self._out.write_log(self.log_step())
        self._out.write_mat(self.h5_dict())
        self._out.write_xyz(self.xyz_string())

    def bind_timers(self, timer: list[str]):
        dic = {
            "est": ("run_est", "EST"),
            "tot": ("propagate", "Total"),
            "sav": ("save_step", "Saving"),
            "wrt": ("write_outputs", "Writing"),
            "qua": ("update_quantum", "Quantum"),
        }
        for key in timer:
            self.add_timer(*dic[key])

    def add_timer(self, name: str, label: str):
        setattr(self.__class__, name, record_time(getattr(self.__class__, name), self._out, label))

    def log_header(self):
        outstr = f"{self._type} dynamics of {self._name}\n"
        return outstr

    def log_step(self):
        outstr = f"\nStep {self._step}\nTime {self._time}\n"
        return outstr

    def dat_header(self, record):
        dic = {}
        dic["time"] = " " + Printer.write("Time [fs]", "s")
        for rec in record:
            dic[rec] = ""
            if rec == "pop":
                for s in range(self.n_states):
                    dic[rec] += Printer.write(f'{s} Population', "s")
            if rec == "pes":
                for s in range(self.n_states):
                    dic[rec] += Printer.write(f'{s} Pot En [eV]', "s")
            if rec == "ken":
                dic[rec] += Printer.write('Total Kin En [eV]', "s")
            if rec == "pen":
                dic[rec] += Printer.write('Total Pot En [eV]', "s")
            if rec == "ten":
                dic[rec] += Printer.write('Total En [eV]', "s")
            if rec == "nacdr":
                for s1 in range(self.n_states):
                    for s2 in range(s1):
                        dic[rec] += Printer.write(f'{s2}-{s1} NACdr [au]', "s")
            if rec == "nacdt":
                for s1 in range(self.n_states):
                    for s2 in range(s1):
                        dic[rec] += Printer.write(f'{s2}-{s1} NACdt [au]', "s")
            if rec == "coeff":
                for s in range(self.n_states):
                    dic[rec] += Printer.write(f'{s} State Coeff', f" <{Printer.field_length*2+1}")
        return dic

    def dat_dict(self, record):
        dic = {}
        dic["time"] = Printer.write(self.curr_time * Constants.au2fs, "f")
        for rec in record:
            dic[rec] = ""
            if rec == "pop":
                for s in range(self.n_states):
                    dic[rec] += Printer.write(np.abs(self.mol.pes.coeff_s[s])**2, "f")
            if rec == "pes":
                for s in range(self.n_states):
                    dic[rec] += Printer.write(self.mol.pes.ham_eig_ss[s,s] * Constants.eh2ev, "f")
            if rec == "ken":
                dic[rec] += Printer.write(self.mol.kinetic_energy * Constants.eh2ev, "f")
            if rec == "pen":
                dic[rec] += Printer.write(self.potential_energy(self.mol) * Constants.eh2ev, "f")
            if rec == "ten":
                dic[rec] += Printer.write(self.total_energy(self.mol) * Constants.eh2ev, "f")
            if rec == "nacdr":
                for s1 in range(self.n_states):
                    for s2 in range(s1):
                        nac = np.sum(self.mol.pes.nacdr_ssad[s1,s2]**2)
                        nac = np.sqrt(nac)
                        dic[rec] += Printer.write(nac, "f")
            if rec == "nacdt":
                for s1 in range(self.n_states):
                    for s2 in range(s1):
                        dic[rec] += Printer.write(self.mol.pes.nacdt_ss[s1,s2], "f")
            if rec == "coeff":
                for s in range(self.n_states):
                    dic[rec] += Printer.write(self.mol.pes.coeff_s[s], "z")
        return dic

    def xyz_string(self):
        return self.mol.to_xyz()

    def vxyz_string(self):
        return self.mol.to_vxyz()

    def h5_info(self):
        mol = self.mol
        to_write = {
            "step": "info",
            "nst": mol.n_states,
            "nat": mol.n_atoms,
            "ats": mol.name_a,
            "mass": mol.mass_a,
            # add others if necessary
        }
        return to_write

    def h5_dict(self):
        mol = self.mol
        to_write = {
            "step": self._step,
            "time": self.curr_time,
            "pos": mol.pos_ad,
            "vel": mol.vel_ad,
            "acc": mol.acc_ad,
            "trans": mol.pes.trans_ss,
            "hdiag": mol.pes.ham_eig_ss,
            "grad": mol.pes.grad_sad,
            "nacdr": mol.pes.nacdr_ssad,
            "nacdt": mol.pes.nacdt_ss,
            "coeff": mol.pes.coeff_s
        }
        return to_write