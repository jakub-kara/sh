import numpy as np
import matplotlib.pyplot as plt
import sys, os
from classes.constants import units

def make_dirs(root):
    os.chdir(root)
    for i in range(nsamp):
        os.mkdir(f"T{i}")

def get_dirs(loc=".", cond=lambda x : True):
    return [d for d in os.listdir(loc) if (os.path.isdir(d) and not d.startswith(".") and cond(d))]

def sample_nuclear():
    gamma = 0.5
    q0 = np.array([-2,0])
    v0 = np.array([0.005, 0])
    m = 1 / units["amu"]
    p0 = m * v0

    q = np.random.normal(q0, 1/np.sqrt(2*gamma), (nsamp, q0.shape[0]))
    p = np.random.normal(p0, np.sqrt(2*gamma), (nsamp, q0.shape[0]))
    v = p / m

    for i in range(nsamp):
        os.chdir(f"T{i}")
        with open("geom.xyz", "w") as f:
            f.write("1\n\n")
            f.write(f"H {q[i,0]} {q[i,1]} 0 {v[i,0]} {v[i,1]} 0")
        os.chdir("..")

def sample_quantum():
    nst = 3
    theta = np.arccos(np.random.uniform(0, 1, (nsamp, nst - 1)))
    phi = np.random.uniform(0, 2*np.pi, (nsamp, nst - 1))

    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    sz = np.cos(theta)

    for i, d in enumerate(get_dirs(".", lambda x: x.startswith("T"))):
        os.chdir(d)
        with open("bloch.dat", "w") as f:
            for j in range(nst - 1):
                f.write(f"{sx[i,j]} {sy[i,j]} {sz[i,j]}\n")
        os.chdir("..")

nsamp = 1000
# make_dirs(".")
# sample_nuclear()
sample_quantum()
