import sys, os
import numpy as np

def get_dirs():
    return [d for d in os.listdir() if (os.path.isdir(d) and not d.startswith("."))]

if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

# ONLY FOR CONSTANT TIMESTEP
dirs = get_dirs()
pos = np.zeros(len(dirs))
for i, d in enumerate(dirs):
    print(d)
    with open(f"{d}/0/data/out.xyz", "r") as f:
        data = f.readlines()
        pos[i] = float(data[-1].split()[1])
np.save("pos.npy", pos)