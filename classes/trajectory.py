import numpy as np
import sys, pickle
import time
from .molecule import Molecule
from .out import Printer, Output
from .constants import Constants
from .meta import Singleton
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram
from updaters.nuclear import NuclearUpdater
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater

class Trajectory:
    def __init__(self, *, dynamics, **config):
        self.mols: list[Molecule] = []
        self._dyn: Dynamics = Dynamics(key = dynamics["method"], dynamics=dynamics, **config)

        self.index = None
        self._split = None

        self.bind_components(dynamics=dynamics, **config)

    @property
    def n_steps(self):
        return len(self.mols)

    @property
    def mol(self):
        return self.mols[-1]

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
        self.mols.append(mol)
        return self

    def pop_molecule(self, index: int):
        self.mols.pop(index)
        return self

    def remove_molecule(self, mol: Molecule):
        self.mols.remove(mol)
        return self

    def split_traj(self):
        pass

    def prepare_traj(self):
        out = Output()
        out.open_log()
        out.to_log("../" + sys.argv[1])
        out.write_log("="*40)
        out.write_log(f"Initialising trajectory")
        out.write_log(f"Step:           {self._dyn.curr_step}")
        out.write_log(f"Time:           {self._dyn.curr_time} fs")
        out.write_log()

        self.write_headers()
        t0 = time.time()
        self._dyn.prepare_traj(self.mol)
        out.write_log(f"Total time:     {time.time() - t0} s")
        out.write_log("="*40)
        out.write_log()

    # TODO: find a better way of timing things
    def run_step(self):
        self.write_outputs()
        self.next_step()

        out = Output()
        out.write_log("="*40)
        out.write_log(f"Step:           {self._dyn.curr_step}")
        out.write_log(f"Time:           {self._dyn.curr_time} fs")
        out.write_log()

        t0 = time.time()
        out.write_log(f"Nuclear + EST")
        self.add_molecule(self._dyn.update_nuclear(self.mols))
        out.write_log(f"Wall time:      {time.time() - t0} s")
        out.write_log()

        t1 = time.time()
        out.write_log(f"Quantum")
        self._dyn.update_quantum(self.mols)
        out.write_log(f"Wall time:      {time.time() - t1} s")
        out.write_log()

        t2 = time.time()
        out.write_log(f"Adjust")
        self.pop_molecule(0)
        self._dyn.adjust_nuclear(self.mols)
        out.write_log(f"Wall time:      {time.time() - t2} s")
        out.write_log()

        t3 = time.time()
        out.write_log(f"Saving")
        self.save_step()
        out.write_log(f"Wall time:      {time.time() - t3} s")
        out.write_log()

        out.write_log(f"Total time:     {time.time() - t0} s")
        out.write_log("="*40)
        out.write_log()

    def bind_components(self, *, electronic: dict, nuclear: dict, quantum: dict, output: dict, **config):
        self.bind_est(**electronic)
        self.bind_nuclear_integrator(nuclear["nuc_upd"])
        mol = self.get_molecule(**nuclear)
        self.add_molecule(mol)
        self.bind_tdc_updater(**quantum)
        self.bind_coeff_updater(**quantum)
        self.bind_io(**output)
        self.bind_molecules(**nuclear)

    def get_molecule(self, **nuclear):
        est = ESTProgram()
        return Molecule(key = nuclear.get("pes", None), n_states=est.n_states, **nuclear)

    def bind_molecules(self, **nuclear):
        nupd = NuclearUpdater()
        for _ in range(max(nupd.steps, CoeffUpdater().steps, nuclear.get("keep", 0))):
            self.add_molecule(self.mol.copy_all())

    def bind_est(self, **electronic):
        ESTProgram(key = electronic["program"], **electronic)

    def bind_nuclear_integrator(self, type: str):
        NuclearUpdater(key = type)

    def bind_tdc_updater(self, **quantum):
        TDCUpdater(key = quantum["tdc_upd"], **quantum)

    def bind_coeff_updater(self, **quantum):
        CoeffUpdater(key = quantum["coeff_upd"], **quantum)

    def bind_io(self, **output):
        Output(**output)
        # self.bind_timers(output["timer"])

    def total_energy(self, mol: Molecule):
        return self._dyn.potential_energy(mol) + mol.kinetic_energy

    # These two are quite hacky, could improve
    def save_step(self):
        est = ESTProgram()
        est.backup_wf()

        out = Output()
        out.close_log()
        with open("backup/traj.pkl", "wb") as pkl:
            self._single = Singleton.save()
            pickle.dump(self, pkl)
        out.open_log()

    @staticmethod
    def load_step(file):
        with open(file, "rb") as pkl:
            traj: Trajectory = pickle.load(pkl)
            Singleton.restore(traj._single)
        out = Output()
        out.open_log()
        return traj

    def write_headers(self):
        out = Output()
        out.write_dat(self.dat_header(out.record), "w")
        out.write_mat(self.h5_info(), "w")
        out.write_xyz("", "w")

    def write_outputs(self):
        out = Output()
        out.write_dat(self.dat_dict(out.record))
        out.write_mat(self.h5_dict())
        out.write_xyz(self.xyz_string())

    def next_step(self):
        self._dyn.next_step()

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

        dic = self._dyn.dat_header(dic, record)
        return dic

    def dat_dict(self, record):
        dic = {}
        dic["time"] = Printer.write(self._dyn.curr_time * Constants.au2fs, "f")
        for rec in record:
            dic[rec] = ""
            if rec == "pop":
                for s in range(self.n_states):
                    # dic[rec] += Printer.write(np.abs(self.mol.coeff_s[s])**2, "f")
                    dic[rec] += Printer.write(self._dyn.population(self.mol, s), "f")
            if rec == "pes":
                for s in range(self.n_states):
                    dic[rec] += Printer.write(self.mol.ham_eig_ss[s,s] * Constants.eh2ev, "f")
            if rec == "ken":
                dic[rec] += Printer.write(self.mol.kinetic_energy * Constants.eh2ev, "f")
            if rec == "pen":
                dic[rec] += Printer.write(self._dyn.potential_energy(self.mol) * Constants.eh2ev, "f")
            if rec == "ten":
                dic[rec] += Printer.write(self.total_energy(self.mol) * Constants.eh2ev, "f")
            if rec == "nacdr":
                for s1 in range(self.n_states):
                    for s2 in range(s1):
                        nac = np.sum(self.mol.nacdr_ssad[s1,s2]**2)
                        nac = np.sqrt(nac)
                        dic[rec] += Printer.write(nac, "f")
            if rec == "nacdt":
                for s1 in range(self.n_states):
                    for s2 in range(s1):
                        dic[rec] += Printer.write(self.mol.nacdt_ss[s1,s2], "f")
            if rec == "coeff":
                for s in range(self.n_states):
                    dic[rec] += Printer.write(self.mol.coeff_s[s], "z")

        dic = self._dyn.dat_dict(dic, record)
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
            "step": self._dyn.curr_step,
            "time": self._dyn.curr_time,
            "pos": mol.pos_ad,
            "vel": mol.vel_ad,
            "acc": mol.acc_ad,
            "trans": mol.trans_ss,
            "hdiag": mol.ham_eig_ss,
            "grad": mol.grad_sad,
            "nacdr": mol.nacdr_ssad,
            "nacdt": mol.nacdt_ss,
            "coeff": mol.coeff_s
        }
        return to_write
