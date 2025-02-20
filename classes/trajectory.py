import numpy as np
import sys, pickle
import time
from copy import deepcopy
from .meta import Singleton
from .molecule import Molecule, MoleculeFactory
from .out import Printer, Output
from .constants import convert
from .timestep import Timestep
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram
from updaters.composite import CompositeIntegrator
from updaters.tdc import TDCUpdater
from updaters.coeff import CoeffUpdater

class Trajectory:
    def __init__(self, *, dynamics: dict, **config):
        self.mols: list[Molecule] = []
        config["nuclear"].setdefault("mixins", [])

        self.index = None
        self._backup = dynamics.get("backup", True)
        self.dyn: Dynamics = Dynamics[dynamics["method"]](dynamics=dynamics, **config)
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
    def n_atoms(self):
        return self.mol.n_atoms

    @property
    def is_finished(self):
        return self._timestep.finished

    @property
    def dt(self):
        return self._timestep.dt

    @property
    def curr_step(self):
        return self._timestep.step

    @property
    def curr_time(self):
        return self._timestep.time

    @property
    def en_thresh(self):
        return self.en_thresh

    def next_step(self):
        self._timestep.next_step()

    def add_molecule(self, mol: Molecule):
        self.mols.append(mol)
        return self

    def pop_molecule(self, index: int):
        self.mols.pop(index)
        return self

    def remove_molecule(self, mol: Molecule):
        self.mols.remove(mol)
        return self

    @property
    def split(self):
        return self.dyn.split

    def split_traj(self):
        clone = deepcopy(self)
        self.mols[-1], clone.mols[-1] = self.dyn.split_mol(self.mol)
        self.dyn.split = None
        clone.dyn.split = None
        return clone

    def prepare_traj(self):
        out = Output()
        out.open_log(mode="w")
        out.to_log("../" + sys.argv[1])
        out.write_border()
        out.write_log(f"Initialising trajectory")
        out.write_log()

        self.write_headers()
        t0 = time.time()
        self.dyn.prepare_traj(self.mol)
        out.write_log(f"Total time:     {time.time() - t0} s")
        out.write_border()
        out.write_log()
        out.close_log()

        self.write_outputs()

    # TODO: find a better way of timing things
    def run_step(self):
        out = Output()
        out.open_log()
        out.write_border()
        out.write_log(f"Step:           {self.curr_step}")
        out.write_log(f"Time:           {convert(self.curr_time, 'au', 'fs'):.4f} fs")
        # out.write_log(f"Time:           {convert(self.curr_time, 'au', 'fs')} fs")
        out.write_log(f"Stepsize:       {convert(self._timestep.dt, 'au', 'fs')} fs")
        out.write_log()

        self.dyn.steps_elapsed(self._timestep.step)

        t0 = time.time()
        # if self._backup:
        t3 = time.time()
        out.write_log(f"Saving")
        self.save_step()
        out.write_log(f"Wall time:      {time.time() - t3} s")
        out.write_log()

        t1 = time.time()
        out.write_log(f"Nuclear + EST")
        print(np.abs(self.mol.coeff_s)**2)
        self.dyn.update_nuclear(self.mols, self._timestep.dt)
        temp = CompositeIntegrator().active.out.out
        print(np.abs(temp.coeff_s)**2)
        valid = self._timestep.validate(self.energy_diff(temp, self.mols))
        if not valid:
            self._timestep.fail()
            ESTProgram().recover_wf()
            return
        self.add_molecule(temp)
        self.pop_molecule(0)

        out.write_log(f"Wall time:      {time.time() - t1} s")
        out.write_log()

        # print()
        # print(f"TOTAL: {self.dyn.total_energy(self.mol)}")
        # print()

        t2 = time.time()
        out.write_log(f"Adjust")
        self.dyn.adjust_nuclear(self.mols, self._timestep.dt)
        out.write_log(f"Wall time:      {time.time() - t2} s")
        out.write_log()

        out.write_log(f"Total time:     {time.time() - t0} s")
        out.write_border()
        out.write_log()
        out.close_log()

        self.next_step()
        self._timestep.success()
        self._timestep.save_nupd()
        self.write_outputs()

        if self.is_finished:
            self.save_step()

    def bind_components(self, *, dynamics: dict, electronic: dict, nuclear: dict, quantum: dict, output: dict, **kwargs):
        self.bind_est(**electronic)
        self.bind_nuclear_integrator(nuclear["nuc_upd"])
        mol = self.get_molecule(**nuclear)
        self.dyn.read_coeff(mol, quantum.get("input", None))
        self.add_molecule(mol)
        self.bind_tdc_updater(**quantum)
        self.bind_coeff_updater(**quantum)
        self.bind_io(**output)
        self.bind_molecules(**nuclear)
        self.bind_timestep(**dynamics)

    def get_molecule(self, **nuclear):
        est = ESTProgram()
        return MoleculeFactory.create_molecule(n_states=est.n_states, **nuclear)

    def bind_molecules(self, **nuclear):
        nupd = CompositeIntegrator()
        for _ in range(max(nupd.steps, CoeffUpdater().steps, nuclear.get("keep", 0))):
            self.add_molecule(self.mol.copy_all())

    def bind_timestep(self, **dynamics):
        self._timestep: Timestep = Timestep[dynamics.get("timestep", "const")](
            steps = len(self.mols), **dynamics)


    # TODO: rework
    def bind_est(self, **electronic):
        ESTProgram.reset()
        ESTProgram[electronic["program"]](**electronic)

    def bind_nuclear_integrator(self, type: str):
        CompositeIntegrator.reset()
        CompositeIntegrator(nuc_upd = type)

    def bind_tdc_updater(self, **quantum):
        TDCUpdater.reset()
        TDCUpdater[quantum["tdc_upd"]](**quantum)

    def bind_coeff_updater(self, **quantum):
        CoeffUpdater.reset()
        CoeffUpdater[quantum["coeff_upd"]](**quantum)

    def bind_io(self, **output):
        Output(**output)

    def energy_diff(self, mol: Molecule, mols: list[Molecule]):
        print(np.abs(self.dyn.total_energy(mol) - self.dyn.total_energy(mols[-1])))
        return np.abs(self.dyn.total_energy(mol) - self.dyn.total_energy(mols[-1]))

    def save_step(self):
        est = ESTProgram()
        est.backup_wf()

        out = Output()
        out.close_log()
        if self._backup:
            with open("backup/traj.pkl", "wb") as pkl:
                pickle.dump(self, pkl)
        out.open_log()

    @staticmethod
    def restart(**config):
        with open("backup/traj.pkl", "rb") as pkl:
            traj: Trajectory = pickle.load(pkl)
        traj.restart_components(**config)

        out = Output()
        out.open_log()
        out.write_log()
        out.write_border()
        out.write_log("Succesfully restarted from backups.")
        out.write_border()
        out.write_log()
        return traj

    def restart_components(self, *, dynamics: dict, electronic: dict, nuclear: dict, quantum: dict, output: dict, **kwargs):
        self._backup = dynamics.get("backup", True)
        self.bind_est(**electronic)
        self.bind_nuclear_integrator(nuclear["nuc_upd"])
        self.bind_tdc_updater(**quantum)
        self.bind_coeff_updater(**quantum)
        # self.bind_io(**output)
        self._timestep.adjust(**dynamics)

    def write_headers(self):
        out = Output()
        out.write_dat(self.dat_header(), "w")
        out.write_h5(self.h5_info(), "w")
        out.write_xyz("", "w")
        out.write_dist("", "w")

    def write_outputs(self):
        out = Output()
        out.write_dat(self.dat_dict())
        out.write_h5(self.h5_dict())
        out.write_xyz(self.vxyz_string())
        out.write_dist(self.dist_string())

    def copy(self):
        return deepcopy(self)

    def dat_header(self):
        nst = self.mol.n_states

        dic = {}
        dic["time"] = "#" + Printer.write("Time [fs]", "s")

        dic["pop"] = "".join([Printer.write(f"Population {i}", "s") for i in range(nst)])
        dic["pes"] = "".join([Printer.write(f"Pot En {i} [eV]", "s") for i in range(nst)])
        dic["ken"] = Printer.write("Total Kin En [eV]", "s")
        dic["pen"] = Printer.write("Total Pot En [eV]", "s")
        dic["ten"] = Printer.write("Total En [eV]", "s")
        dic["nacdr"] = "".join([Printer.write(f"NACdr {j}-{i} [au]", "s") for i in range(nst) for j in range(i)])
        dic["nacdt"] = "".join([Printer.write(f"NACdr {j}-{i} [au]", "s") for i in range(nst) for j in range(i)])
        dic["coeff"] = "".join([Printer.write(f"Coeff {i}", f" <{Printer.field_length*2+1}") for i in range(nst)])
        dic["posx"] = Printer.write("X-Position [au]", "s")
        dic["posy"] = Printer.write("Y-Position [au]", "s")
        dic["posz"] = Printer.write("Z-Position [au]", "s")
        dic["momx"] = Printer.write("X-Momentum [au]", "s")
        dic["momy"] = Printer.write("Y-Momentum [au]", "s")
        dic["momz"] = Printer.write("Z-Momentum [au]", "s")

        dic |= self.dyn.dat_header()
        return dic

    def dat_dict(self):
        nst = self.mol.n_states
        mol = self.mol
        dyn = self.dyn

        dic = {}
        dic["time"] = Printer.write(convert(self.curr_time, "au", "fs"), "f")

        dic["pop"] = "".join([Printer.write(dyn.population(self.mol, i), "f") for i in range(nst)])
        dic["pes"] = "".join([Printer.write(convert(mol.ham_eig_ss[i,i], "au", "ev"), "f") for i in range(nst)])
        dic["ken"] = Printer.write(convert(mol.kinetic_energy, "au", "ev"), "f")
        dic["pen"] = Printer.write(convert(dyn.potential_energy(self.mol), "au", "ev"), "f")
        dic["ten"] = Printer.write(convert(dyn.total_energy(self.mol), "au", "ev"), "f")
        dic["nacdr"] = "".join([Printer.write(mol.nac_norm_ss[i,j], "f") for i in range(nst) for j in range(i)])
        dic["nacdt"] = "".join([Printer.write(mol.nacdt_ss[i,j], "f") for i in range(nst) for j in range(i)])
        dic["coeff"] = "".join([Printer.write(mol.coeff_s[i], "z") for i in range(nst)])
        dic["posx"] = Printer.write(mol.pos_ad[0,0], "f")
        dic["posy"] = Printer.write(mol.pos_ad[0,1], "f")
        dic["posz"] = Printer.write(mol.pos_ad[0,2], "f")
        dic["momx"] = Printer.write(mol.mom_ad[0,0], "f")
        dic["momy"] = Printer.write(mol.mom_ad[0,1], "f")
        dic["momz"] = Printer.write(mol.mom_ad[0,2], "f")

        dic |= dyn.dat_dict()
        return dic

    def dist_string(self):
        return self.mol.to_dist()

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
        to_write = self.dyn.h5_dict()
        to_write.update({
            "step": self.curr_step,
            "time": self.curr_time,
            "pos": mol.pos_ad,
            "vel": mol.vel_ad,
            "acc": mol.acc_ad,
            "trans": mol.trans_ss,
            "hdiag": mol.ham_eig_ss,
            "grad": mol.grad_sad,
            "nacdr": mol.nacdr_ssad,
            "nacdt": mol.nacdt_ss,
            "coeff": mol.coeff_s
        })
        return to_write
