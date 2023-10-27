import sys
import pickle

from classes import TrajectorySH, SimulationSH
from propagators import initialise_dynamics, loop_dynamics
from io_methods import finalise_dynamics
from utility import file_to_dictionary

def main():
    if len(sys.argv) > 1:
        input_dict = file_to_dictionary(sys.argv[1])
        ctrl = SimulationSH(input_dict)
        traj = TrajectorySH(input_dict)

        initialise_dynamics(traj, ctrl)
    else:
        with open("backup/traj.pkl", "rb") as traj_pkl:
            traj = pickle.load(traj_pkl)
        with open("backup/ctrl.pkl", "rb") as ctrl_pkl:
            ctrl = pickle.load(ctrl_pkl)
    
    loop_dynamics(traj, ctrl)
    finalise_dynamics(traj)
    
if __name__ == "__main__":
    main()