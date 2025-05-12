import numpy as np
import os, sys, shutil
import pickle
from copy import deepcopy
from .meta import Singleton, Factory
from classes.trajectory import Trajectory
from classes.out import Output as out

class Bundle:
    def __init__(self, **config: dict):
        self.trajs: list[Trajectory] = []
        self.iactive: int = 0
        self.active: Trajectory = None

        self.config = config

    @property
    def n_traj(self):
        return len(self.trajs)

    def save_setup(self):
        with open("single.pkl", "wb") as pkl:
            pickle.dump(Singleton.save(), pkl)

        with open("out.pkl", "wb") as pkl:
            pickle.dump(out.save(), pkl)

        with open("fact.pkl", "wb") as pkl:
            pickle.dump(Factory.save(), pkl)

    @staticmethod
    def load_setup():
        with open("single.pkl", "rb") as pkl:
            Singleton.restart(pickle.load(pkl))

        with open("out.pkl", "rb") as pkl:
            out.restart(pickle.load(pkl))

        with open("fact.pkl", "rb") as pkl:
            Factory.restart(pickle.load(pkl))

    @staticmethod
    def restart(**config):
        bundle = Bundle(**config)
        traj_dirs = [d for d in os.listdir() if (os.path.isdir(d) and d.isdigit())]
        for d in traj_dirs:
            os.chdir(d)
            traj = Trajectory.restart(**config)
            bundle.add_trajectory(traj)
            os.chdir("..")
        return bundle

    def add_trajectory(self, traj: Trajectory):
        traj.index = self.n_traj
        self.trajs.append(traj)
        return self

    def set_active(self):
        self.iactive = np.argmin([traj.timestep.time for traj in self.trajs])
        self.active = self.trajs[self.iactive]
        return self

    @property
    def is_finished(self):
        return np.all([traj.is_finished for traj in self.trajs])

    def edit(self, attr, val):
        for traj in self.trajs:
            setattr(traj, attr, val)