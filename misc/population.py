import matplotlib.pyplot as plt
import numpy as np
from utility import get_dirs, get_ext
import sys, os

fig, ax = plt.subplots()
orig = os.getcwd()
for parent in sys.argv[1:]:
    print(parent)
    os.chdir(f"{parent}")
    dirs = get_dirs()
    t = []
    pop = [[],[]]
    c = 0
    for d, dir in enumerate(dirs):
        os.chdir(f"{dir}/data")
        exts = get_ext("dat")
        for e, ext in enumerate(exts):
            c += 1
            with open(f"{ext}", "r") as file:
                for i, line in enumerate(file):
                    if i == 0: continue
                    data = line.strip().split()
                    if d == 0 and e == 0: 
                        t.append(float(data[0]))
                        pop[0].append(0)
                        pop[1].append(0)
                    pop[int(data[1])][i-1] += 1
        os.chdir("../..")
    ax.plot(t, [i/c for i in pop[1]])
    os.chdir(f"{orig}")

plt.show()
