import numpy as np
import sys, os
import h5py

def get_dirs(loc="."):
    return [d for d in os.listdir(loc) if (os.path.isdir(d) and d.isdigit())]

class Frame:
    def __init__(self):
        self.weight = 0
        self.coeff = None

# hardcoded filenames
class Traj:
    def __init__(self):
        self.steps: dict[float, Frame] = {}
        
    @property
    def n_step(self):
        return len(self.steps)
    
    def make_frames(self):
        with h5py.File("data/out.h5", "r") as file:
            for step in file.keys():
                time = file[f"{step}/time"][()]
                self.steps[time] = Frame()
        return self   

    def read_coeff(self):
        with h5py.File("data/out.h5", "r") as file:
            for time in file.keys():
                coeff = file[f"{time}/coeff"][:]
                self.steps[int(time)].coeff = np.array(coeff)
        return self
    
    def get_pop(self, time):
        step = self.steps[time]
        return np.abs(step.coeff)**2 * np.abs(step.weight)**2

class Bundle:
    def __init__(self):
        self.trajs: dict[str, Traj] = {}
    
    @property
    def n_traj(self):
        return len(self.trajs)

    def make_trajs(self):
        drs = get_dirs()
        for dr in drs:
            os.chdir(dr)
            self.trajs[dr] = Traj().make_frames()
            os.chdir("..")
        return self

    def set_weight(self, weight):
        clones = {}
        with open("clone.log", "r") as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                data = line.strip().split()
                clones[float(data[0])] = (int(data[1]), int(data[2]))
        
        for frm in self[0].children.values():
            frm.weight = weight
        
        for time, (par, chl) in clones.items():
            state = np.argmin(self[par].children.values())
            for frm in self[0].children.values():
                frm.weight = 1

    def read_coeff(self):
        for key, traj in self.children.items():
            os.chdir(key)
            traj.read_coeff()
            os.chdir("..")
        return self

class Ensemble:
    def __init__(self, root):
        os.chdir(root)
        self.bunds: dict[str, Bundle] = {}

    @property
    def n_bund(self):
        return len(self.bunds)

    def make_bunds(self):
        drs = get_dirs()
        for dr in drs:
            os.chdir(dr)
            self.bunds[dr] = Bundle().make_trajs()
            os.chdir("..")
        return self
    
    def set_weight(self):
        weight = 1/self.n_bund
        for dr, bund in self.bunds.items():
            os.chdir(dr)
            bund.set_weight(weight)
            os.chdir("..")

    def read_coeff(self):
        for key, bund in self.bunds.items():
            os.chdir(key)
            bund.read_coeff()
            os.chdir("..")
        return self
    
    def get_pop(self):
        pop = np.zeros(self[0][0][0].coeff.shape)

        return pop

ens = Ensemble(sys.argv[1]).make_bunds().read_coeff()
breakpoint()