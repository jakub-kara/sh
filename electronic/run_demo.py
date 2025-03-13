import numpy as np
import sys
from classes.molecule import Molecule
from classes.constants import convert

def main():
    a = 0.005
    b = 6
    d = 0.2

    states = 0
    grads = []
    nacs = []

    with open(sys.argv[1], "r") as f:
        for line in f:
            if line.startswith("geom"):
                geom = line.split()[1]
            if line.startswith("states"):
                states = int(line.split()[1])
            if line.startswith("grad"):
                grads.append(int(line.split()[1]))
            if line.startswith("nac"):
                nacs.append([int(line.split()[1]), int(line.split()[2])])

    mol = Molecule(n_states= states, input = geom, vxyz = False)
    mol.pos_ad = convert(mol.pos_ad, "aa", "au")
    gradham = np.zeros((mol.n_states, mol.n_states, mol.n_atoms, mol.n_dim))
    mol.ham_dia_ss[:] = a*d
    for at in range(mol.n_atoms - 1):
        dx = mol.pos_ad[at] - mol.pos_ad[at+1]
        x = np.linalg.norm(dx)
        mol.ham_dia_ss[0,0] += a*(x - b)**2
        gradham[0,0,at] += 2*a*(x - b) * dx/x
        gradham[0,0,at+1] -= 2*a*(x - b) * dx/x
        for i in range(1, mol.n_states):
            mol.ham_dia_ss[i,i] += a*(1/i * (x - b - np.sqrt(i))**2 + i)
            gradham[i,i,at] += a*(2/i * (x - b - np.sqrt(i)) * dx/x)
            gradham[i,i,at+1] -= a*(2/i * (x - b - np.sqrt(i)) * dx/x)

    mol.ham_eig_ss, mol.trans_ss = diagonalise(mol.ham_dia_ss)
    mol.grad_sad, mol.nacdr_ssad = get_grad_nac(gradham, mol.ham_eig_ss, mol.trans_ss)

    np.save("ham.npy", mol.ham_eig_ss)
    gradout = np.zeros((mol.n_states, mol.n_atoms, mol.n_dim))
    for grad in grads:
        gradout[grad] = np.real(mol.grad_sad[grad])
    np.save("grad.npy", gradout)

    nacout = np.zeros((mol.n_states, mol.n_states, mol.n_atoms, mol.n_dim))
    for (i,j) in nacs:
        nacout[i,j] = np.real(mol.nacdr_ssad[i,j])
        nacout[j,i] = -nacout[i,j]
    np.save("nac.npy", nacout)

def diagonalise(hamdia):
    eval, evec = np.linalg.eigh(hamdia)
    return np.diag(eval), evec

def get_grad_nac(gradham, hameig, trans):
    nstates = trans.shape[0]
    grad = np.einsum("is, ijad, js -> sad", trans.conj(), gradham, trans)
    temp = np.einsum("is, ijad, jr -> srad", trans.conj(), gradham, trans)
    nac = temp * (1 - np.eye(nstates)[:, :, None, None])
    for s1 in range(nstates):
        for s2 in range(nstates):
            if s1 == s2: continue
            nac[s1,s2] /= hameig[s2,s2] - hameig[s1,s1]
    return grad, nac

if __name__ == "__main__":
    main()