import numpy as np
import os
from constants import Constants

class LVC:
    freqfile = f"{os.getcwd()}/molcas.freq.molden"
    with open(freqfile, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            if "[N_FREQ]" in line:
                line = file.readline()
                nat = (int(line)+6)//3
                freq = np.zeros(3*nat)
                mass = np.zeros(3*nat)
                ref = np.zeros(3*nat)
                name = np.full(nat, "00")
                mat = np.zeros((3*nat, 3*nat))

            if "[FREQ]" in line:
                for i in range(6,3*nat):
                    line = file.readline()
                    freq[i] = float(line)/Constants.eh2cm
            
            if "[FR-COORD]" in line:
                for i in range(nat):
                    line = file.readline()
                    data = line.strip().split()
                    name[i] = data[0]
                    mass[3*i:3*i+3] = Constants.atomic_masses[data[0]]
                    for j in range(3): ref[3*i+j] = float(data[j+1])

            if "vibration" in line:
                v = int(line.strip().split()[-1]) + 5
                for i in range(nat):
                    line = file.readline()
                    data = line.strip().split()
                    for j in range(3): mat[v, 3*i+j] = float(data[j])
    
    paramfile = f"{os.getcwd()}/LVC.template"
    with open(paramfile, 'r') as file:
        line = file.readline()
        line = file.readline()
        n = np.array([int(i) for i in line.strip().split()])
        nst = np.sum(n)
        off = np.cumsum(n)-n
        epsilon = np.zeros(nst)
        kappa = np.zeros((nst,nat*3))
        lamda = np.zeros((nst,nst,nat*3))
        eta = np.zeros((nst,nst))

        while True:
            line = file.readline()
            if not line:
                break

            if "epsilon" in line:
                line = file.readline()
                tot = int(line.strip())
                for i in range(tot):
                    line = file.readline()
                    data = line.strip().split()
                    plet = int(data[0])
                    st = int(data[1])
                    val = float(data[2])
                    epsilon[off[plet-1]+st-1] = val
            
            if "kappa" in line:
                line = file.readline()
                tot = int(line.strip())
                for i in range(tot):
                    line = file.readline()
                    data = line.strip().split()
                    plet = int(data[0])
                    st = int(data[1])
                    mode = int(data[2])
                    val = float(data[3])
                    kappa[off[plet-1]+st-1, mode] = val
            
            if "lambda" in line:
                line = file.readline()
                tot = int(line.strip())
                for i in range(tot):
                    line = file.readline()
                    data = line.strip().split()
                    plet = int(data[0])
                    st1 = int(data[1])
                    st2 = int(data[2])
                    mode = int(data[3])
                    val = float(data[4])
                    lamda[off[plet-1]+st1-1, off[plet-1]+st2-1, mode] = val
                    lamda[off[plet-1]+st2-1, off[plet-1]+st1-1, mode] = -val

    xyzfile = f"{os.getcwd()}/ref.xyz"
    with open(xyzfile, 'w') as file:
        vel = np.zeros_like(ref)
        file.write(f"{nat}\n")
        file.write("\n")
        for i in range(nat):
            file.write(f"{name[i]} {ref[3*i]} {ref[3*i+1]} {ref[3*i+2]} {vel[3*i]} {vel[3*i+1]} {vel[3*i+2]}\n")
    
    @staticmethod
    def cart_to_freq(cart_coords: np.ndarray):
        displ = cart_coords - LVC.ref
        freq_coords = np.zeros(3*LVC.nat)
        for i in range(3*LVC.nat):
            for j in range(3*LVC.nat):
                freq_coords[i] += LVC.mat[j,i]*np.sqrt(LVC.mass[j])*displ[j]
            freq_coords[i] *= np.sqrt(LVC.freq[i])
        return freq_coords
    
    @staticmethod
    def freq_to_cart(freq_coords: np.ndarray):
        cart_coords = np.zeros(3*LVC.nat)
        for i in range(3*LVC.nat):
            for j in range(3*LVC.nat):
                cart_coords[i] += LVC.mat[j,i]/np.sqrt(LVC.mass[j])*freq_coords[j]
            cart_coords[i] /= np.sqrt(LVC.freq[i])
        return cart_coords
    
    @staticmethod
    def get_est(coords: np.ndarray):
        diab = LVC.hamiltonian(coords)
        gradh = LVC.gradient(coords)
        diag, state = LVC.diagonalise_hamiltonian(diab)
        nac = LVC.get_nac(diag, gradh, state)
        return diab, diag, state, nac.reshape((LVC.nst, LVC.nst, LVC.nat, 3))

    @staticmethod
    def hamiltonian(coords: np.ndarray):
        ham = np.zeros((LVC.nst, LVC.nst))
        for i in range(LVC.nst):
            for j in range(LVC.nst):
                if i == j:
                    v0 = 0.5*np.sum(LVC.freq*coords**2)
                    w = LVC.epsilon[i] + np.sum(LVC.kappa[i]*coords)
                    ham[i,i] = v0 + w
                else:
                    ham[i,j] = np.sum(LVC.lamda[i,j]*coords)
        return ham

    @staticmethod
    def gradient(coords: np.ndarray):
        gradh = np.zeros((LVC.nst, LVC.nst, LVC.nat*3))
        for i in range(LVC.nst):
            for j in range(LVC.nst):
                if i == j:
                    gradh[i,i] = LVC.freq*coords + LVC.kappa[i]
                else:
                    gradh[i,j] = LVC.lamda[i,j]
        return gradh
    
    @staticmethod
    def diagonalise_hamiltonian(diab: np.ndarray):
        eval, evec = np.linalg.eigh(diab)
        diag = np.diag(eval)
        return diag, evec
    
    @staticmethod
    def get_nac(diag, gradh, state):
        nac = np.zeros((LVC.nst, LVC.nst, LVC.nat*3))
        for i in range(LVC.nst):
            for j in range(LVC.nst):
                for k in range(LVC.nat*3):
                    nac[i,j,k] = np.conj(state[:,i]) @ gradh[:,:,k] @ state[:,j]
                if i != j:
                    nac[i,j] /= diag[j,j] - diag[i,i]
        return nac