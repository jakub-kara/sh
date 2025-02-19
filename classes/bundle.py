import numpy as np
import os, shutil
import pickle
from .meta import Singleton
from .molecule import MoleculeFactory
from classes.trajectory import Trajectory

class Bundle:
    def __init__(self):
        self._trajs: list[Trajectory] = []
        self._iactive: int = 0
        self._active: Trajectory = None

    def save_setup(self):
        with open("single.pkl", "wb") as pkl:
            pickle.dump(Singleton.save(), pkl)

        with open("mol.pkl", "wb") as pkl:
            pickle.dump(MoleculeFactory.save(), pkl)

    @staticmethod
    def load_setup():
        with open("single.pkl", "rb") as pkl:
            Singleton.restart(pickle.load(pkl))

        with open("mol.pkl", "rb") as pkl:
            MoleculeFactory.restart(pickle.load(pkl))

    @staticmethod
    def restart(**config):
        bundle = Bundle()
        traj_dirs = [d for d in os.listdir() if (os.path.isdir(d) and d.isdigit())]
        Bundle.load_setup()
        for d in traj_dirs:
            os.chdir(d)
            traj = Trajectory.restart(**config)
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
        self._iactive = np.argmin([traj.curr_time for traj in self._trajs])
        self._active = self._trajs[self._iactive]
        return self

    def setup(self, **config):
        traj = Trajectory(**config)
        self.save_setup()
        self.add_trajectory(traj)
        with open("events.log", "w") as f:
            f.write(f"INIT 0\n")
        self.prepare_trajs()
        return self

    def prepare_trajs(self):
        for traj in self._trajs:
            os.chdir(f"{traj.index}")
            traj.prepare_traj()
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
        if self._active.split:
            shutil.copytree(f"{self._iactive}", f"{self.n_traj}", dirs_exist_ok=True)
            with open("events.log", "a") as f:
                temp = np.sum(np.abs(self._active.mol.coeff_s[self._active.split])**2)
                f.write(f"CLONE {self._iactive} {np.sqrt(temp)} {self.n_traj} {np.sqrt(1-temp)} {self._active.curr_step} {self._active.curr_time:.4f}\n")
            clone = self._active.split_traj()
            self.add_trajectory(clone)
        return self

    @property
    def is_finished(self):
        return np.all([traj.is_finished for traj in self._trajs])

    def edit(self, attr, val):
        for traj in self._trajs:
            setattr(traj, attr, val)