import numpy as np
import os, shutil
from classes.trajectory import TrajectoryDynamics
from classes.out import Printer
from classes.constants import Constants
from dynamics import select_dynamics


class Bundle:
    def __init__(self):
        self._trajs: list[TrajectoryDynamics] = []
        self._iactive: int = 0
        self._active: TrajectoryDynamics = None

    @staticmethod
    def from_pkl():
        bundle = Bundle()
        traj_dirs = [d for d in os.listdir() if (os.path.isdir(d) and d.isdigit())]
        for d in traj_dirs:
            traj = TrajectoryDynamics.load_step(f"{d}/backup/traj.pkl")
            bundle.add_trajectory(traj)
        return bundle

    @property
    def n_traj(self):
        return len(self._trajs)

    def add_trajectory(self, traj: TrajectoryDynamics):
        traj.index = self.n_traj
        self._trajs.append(traj)
        return self

    def set_active(self):
        self._iactive = np.argmin([traj.curr_time for traj in self._trajs])
        self._active = self._trajs[self._iactive]
        return self

    def setup_dirs(self, index):
        if os.path.isdir(f"{index}"):
            shutil.rmtree(f"{index}")
        os.mkdir(f"{index}")
        os.mkdir(f"{index}/backup")
        os.mkdir(f"{index}/data")
        os.mkdir(f"{index}/est")
        # HARDCODED
        os.system(f"cp molpro.wf {index}/est")
        return self

    def setup(self, config: dict):
        self.setup_dirs(0)
        with open("clone.log", "w") as f:
            f.write(" " + Printer.write("Time [fs]", "s") + Printer.write("States", "s"))
            f.write(Printer.write("Parent", "s") + Printer.write("|Amplitude|", "s"))
            f.write(Printer.write("Child", "s") + Printer.write("|Amplitude|", "s"))
            f.write(Printer.write("Total", "s") + "\n")

        traj = select_dynamics(config["dynamics"]["method"])(**config)
        self.add_trajectory(traj)
        self.prepare_trajs()
        return self

    def prepare_trajs(self):
        for traj in self._trajs:
            os.chdir(f"{traj.index}")
            traj.prepare_traj()
            traj.write_outputs()
            traj.next_step()
            os.chdir("..")

    def run_step(self):
        self.set_active()
        print()
        if self.n_traj > 1:
            print(self._iactive, self.n_traj)

        os.chdir(f"{self._iactive}")
        self._active.propagate()
        self._active.write_outputs()
        self._active.save_step()
        self._active.next_step()
        os.chdir("..")

        # TODO: implement multiple states
        if self._active.split_states:
            shutil.copytree(f"{self._iactive}", f"{self.n_traj}", dirs_exist_ok=True)
            with open("clone.log", "a") as f:
                temp = np.sqrt(np.sum(np.abs(self._active.mol.pes.coeff_s[self._active.split_states])**2))
                f.write(Printer.write(self._active.curr_time * Constants.au2fs, "f") + Printer.write(" " + ", ".join(map(str, self._active.split_states)), "s"))
                f.write(Printer.write(self._iactive, "i") + Printer.write(np.sqrt(1 - temp**2), "f"))
                f.write(Printer.write(self.n_traj, "i") + Printer.write(temp, "f"))
                f.write(Printer.write(self.n_traj + 1, "i") + "\n")

            clone = self._active.split_traj()
            self.add_trajectory(clone)

        return self

    @property
    def is_finished(self):
        return np.all([traj.is_finished for traj in self._trajs])

    def edit(self, attr, val):
        for traj in self._trajs:
            setattr(traj, attr, val)