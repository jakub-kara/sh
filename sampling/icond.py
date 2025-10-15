import numpy as np
import sys, os
import json
from classes.molecule import Molecule
from electronic.base import ESTProgram, ESTMode, HamTransform


def icond(*, nuclear, electronic, **kwargs):
    est: ESTProgram = ESTProgram.select(electronic["program"])(**electronic)
    mol = Molecule(**nuclear, n_states = est.n_states)
    HamTransform.select(electronic.get("transform", "none"))()

    est.request(ESTMode("d")())
    est.run(mol)
    est.read(mol)

    np.save('energy.npy', mol.ham_eig_ss)
    np.save('dipole.npy', mol.dipmom_ssd)

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        config = json.load(f)
    os.mkdir("est")
    icond(**config)
