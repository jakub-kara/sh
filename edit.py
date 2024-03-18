import sys
import pickle
from classes import Trajectory

with open(sys.argv[1], "rb") as f: traj: Trajectory = pickle.load(f)
traj.ctrl.en_thresh[0] = float(sys.argv[2])
traj.ctrl.en_thresh[1] = float(sys.argv[3])
with open(sys.argv[1], "wb") as f: pickle.dump(traj, f)