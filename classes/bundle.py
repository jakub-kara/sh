import numpy as np
import os, shutil
from classes.trajectory import Trajectory
from classes.out import Printer
from classes.constants import Constants

class Bundle:
    def __init__(self):
        self._trajs: list[Trajectory] = []
        self._iactive: int = 0
        self._active: Trajectory = None

    @staticmethod
    def from_pkl():
        bundle = Bundle()
        traj_dirs = [d for d in os.listdir() if (os.path.isdir(d) and d.isdigit())]
        for d in traj_dirs:
            os.chdir(d)
            traj = Trajectory.load_step(f"backup/traj.pkl")
            bundle.add_trajectory(traj)
            os.chdir("..")
        return bundle

    @property
    def n_traj(self):
        return len(self._trajs)

    def add_trajectory(self, traj: Trajectory):
        traj.index = self.n_traj
        self._trajs.append(traj)
        return self

    def set_active(self):
        self._iactive = np.argmin([traj.dyn.curr_time for traj in self._trajs])
        self._active = self._trajs[self._iactive]
        return self

    def setup(self, config: dict):
        traj = Trajectory(**config)
        self.add_trajectory(traj)
        with open("events.log", "w") as f:
            f.write(f"Trajectory 0 initiated at time = {traj.dyn.curr_time}")
        self.prepare_trajs()
        return self

    def prepare_trajs(self):
        for traj in self._trajs:
            os.chdir(f"{traj.index}")
            traj.prepare_traj()
            traj.dyn.next_step()
            traj.dyn._timestep.success()
            traj.write_outputs()
            if traj._backup:
                traj.save_step()
            os.chdir("..")

    def run_step(self):
        self.set_active()
        print()
        if self.n_traj > 1:
            print(self._iactive, self.n_traj)

        os.chdir(f"{self._iactive}")
        self._active.run_step()
        os.chdir("..")

        # TODO: implement multiple states
        if self._active.split_states:
            shutil.copytree(f"{self._iactive}", f"{self.n_traj}", dirs_exist_ok=True)
            with open("events.log", "a") as f:
                f.write(f"Trajectory {self._iactive} cloned to {self.n_traj}")

            clone = self._active.split_traj()
            self.add_trajectory(clone)

        return self

    @property
    def is_finished(self):
        return np.all([traj.dyn.is_finished for traj in self._trajs])

    def edit(self, attr, val):
        for traj in self._trajs:
            setattr(traj, attr, val)