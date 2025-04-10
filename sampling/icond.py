import numpy as np
import sys, os
from classes.molecule import Molecule
from electronic.base import ESTProgram, ESTMode, HamTransform


def icond(*, nuclear, electronic, **kwargs):
    est: ESTProgram = ESTProgram[electronic["program"]](**electronic)
    mol = Molecule(**nuclear, n_states = est.n_states)
    HamTransform[electronic.get("transform", "none")]()

    est.request(ESTMode("d")())
    est.run(mol)
    est.read(mol)

    np.save('energy.npy', mol.ham_eig_ss)
    np.save('dipole.npy', mol.dipmom_ssd)
