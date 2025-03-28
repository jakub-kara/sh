import sys, os
import numpy as np

def get_dirs():
    return [d for d in os.listdir() if (os.path.isdir(d) and not d.startswith("."))]

if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

# ONLY FOR CONSTANT TIMESTEP
dirs = get_dirs()
for i, d in enumerate(dirs):
    if i == 0:
        with open(f"{d}/0/data/out.dat", "r") as f:
            for iline, line in enumerate(f):
                if iline == 0:
                    nst = line.count("Population")

        data = np.genfromtxt(f"{d}/0/data/out.dat", skip_header=1)
        pop = np.zeros((data.shape[0], nst))
        tim = data[:,0]

    print(d)
    data = np.genfromtxt(f"{d}/0/data/out.dat", skip_header=1)[:,1:nst+1]
    pop += data

pop /= len(dirs)
np.savetxt("pop.dat", np.append(tim[:,None], pop, axis=1))