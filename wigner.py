import numpy as np
import re

from constants import Constants
from classes import Geometry

molecule = np.array([
    [1.271898952305, -0.000000000000, 0.000000000000],
    [-1.271898952305, -0.000000000000, 0.000000000000],
    [2.354576358487, 1.764577651763, 0.000000000000],
    [2.354576358487, -1.764577651763, 0.000000000000],
    [-2.354576358487, -1.764577651763, 0.000000000000],
    [-2.354576358487, 1.764577651763, 0.000000000000]])

test_vel = np.array([
    [1,0,0],
    [1,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0]
], dtype=float)

test_vel2 = np.array([
    [0,1,0],
    [0,-1,0],
    [0,0,0],
    [0,0,0],
    [0,0,0],
    [0,0,0]
], dtype=float)

mass_a = np.array([
    Constants.atomic_masses["C"],
    Constants.atomic_masses["C"],
    Constants.atomic_masses["H"],
    Constants.atomic_masses["H"],
    Constants.atomic_masses["H"],
    Constants.atomic_masses["H"]])
mass_a *= Constants.amu


def read_freq(file: str, ndf: int, linear=False):
    if linear:
        nc = ndf - 5
        numcut = 5
    else:
        nc = ndf - 6
        numcut = 6

    var = 1
    str2search = 'Mass Weighted 2nd Derivative Matrix Eigenvalues'
    str2search2 = "Mass Weighted 2nd Derivative Matrix Eigenvectors"
    str2search3 = 'Low Vibration'
    eigenval = []
    eigenvec = []
    ndf_var = 0
    ndf_var2 = 0
    ndf_var3 = 0
    with open(file, 'r') as f:
        for line in f:
            if str2search in line:
                var = 2
                continue
            if var == 2:
                n = re.findall("\d+\.\d+", line)
                if n:
                    eigenval.extend(n)

            if str2search2 in line:
                var = 3
                continue
            if var == 3:
                n = re.findall("(\d*\.\d+|-\d*\.\d+)", line)

                if n:

                    if ndf_var <= ndf - 1:
                        eigenvec.append(n)
                        ndf_var += 1
                    elif ndf_var2 <= ndf - 1:
                        eigenvec[ndf_var2].extend(n)
                        ndf_var2 += 1
                    else:
                        eigenvec[ndf_var3].extend(n)
                        ndf_var3 += 1

            if str2search3 in line:
                var = 1
                continue

    eigenval = np.asarray(eigenval).astype(float)
    eigenvec = np.asarray(eigenvec).astype(float)

    return eigenval[:], eigenvec[:, :]

def wigner_probability(Q, P, temperature=0):
    if temperature == 0:
        n = 0
    else:
        #implement partition function for vibrational excited states
        pass
      
    if n == 0: # vibrational ground state
        return np.exp(-Q**2) * np.exp(-P**2), n
    else:
        pass


def initial_sampling(geo: Geometry, config: dict, freq_file="freq.out"):
    positions = np.zeros((config.ensemble.ntraj.value, geo.n_atoms, 3))
    velocities = np.zeros((config.ensemble.ntraj.value, geo.n_atoms, 3))

    if config.ensemble.wigner.value:
        potential_energy = 0.
        eigenval, eigenvec = read_freq(file=freq_file, ndf=geo.n_atoms*3)
        eigenval = np.sqrt(eigenval/Constants.amu)

        for i in range(config.ensemble.ntraj.value):
            for i in range(6, geo.n_atoms*3): # for each uncoupled harmonatomlist oscillator
                while True:
                    # get random Q and P in the interval [-5,+5]
                    # this interval is good for vibrational ground state
                    # should be increased for higher states
                    # TODO: needs to be restructured: first obtain temperature, then draw random numbers, then compute wigner probability
                    random_Q = np.random.random()*10 - 5
                    random_P = np.random.random()*10 - 5
                    # calculate probability for this set of P and Q with Wigner distr.
                    probability, vib_state = wigner_probability(random_Q, random_P)
                    if probability > np.random.random():
                        break # coordinates accepted
                # now transform the dimensionless coordinate into a real one
                # paper says, that freq_factor is sqrt(2*PI*freq)
                # QM programs directly give angular frequency (2*PI is not needed)
                freq_factor = np.sqrt(eigenval[i])
                # Higher frequencies give lower displacements and higher momentum.
                # Therefore scale random_Q and random_P accordingly:
                random_Q /= freq_factor
                random_P *= freq_factor
                # add potential energy of this mode to total potential energy
                potential_energy += 0.5 * eigenval[i]**2 * random_Q**2
                for a in range(geo.n_atoms): # for each atom
                    for d in range(3): # and each direction
                        # distort geometry according to normal mode movement
                        # and unweigh mass-weighted normal modes
                        positions[i,a,d] += random_Q * eigenvec[i,a*3+d] * np.sqrt(1./geo.mass_a[a])
                        velocities[i,a,d] += random_P * eigenvec[i,a*3+d] * np.sqrt(1./geo.mass_a[a])
    else:
        for i in range(config.ensemble.ntraj.value):
            positions[i] = geo.position_mnad[-1,0]
            velocities[i] = geo.velocity_mnad[-1,0]
    
    return positions, velocities