import numpy as np
import matplotlib.pyplot as plt
import sys, os
from classes.constants import units

def make_dirs(root):
    os.chdir(root)
    for i in range(nsamp):
        os.mkdir(f"T{i}")

def sample_nuclear():
    gamma = 0.5
    q0 = -15
    e0 = 70
    p0 = np.sqrt(2*e0/units["amu"])

    q = np.random.normal(q0, 1/np.sqrt(2*gamma), nsamp)
    p = np.random.normal(p0, np.sqrt(2*gamma), nsamp)
    v = p * units["amu"]

    for i in range(nsamp):
        os.chdir(f"T{i}")
        with open("geom.xyz", "w") as f:
            f.write("1\n\n")
            f.write(f"1 {q[i]} 0 0 {v[i]} 0 0")
        os.chdir("..")

def sample_quantum():
    nst = 3
    theta = np.arccos(np.random.uniform(0, 1, (nsamp, nst - 1)))
    phi = np.random.uniform(0, 2*np.pi, (nsamp, nst - 1))

    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    sz = np.cos(theta)

    for i in range(nsamp):
        os.chdir(f"T{i}")
        with open("bloch.dat", "w") as f:
            for j in range(nst - 1):
                f.write(f"{sx[i,j]} {sy[i,j]} {sz[i,j]}\n")
        os.chdir("..")

nsamp = 2000
make_dirs(".")
sample_nuclear()
sample_quantum()
