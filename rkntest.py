import numpy as np

from classes.molecule import Molecule, MoleculeMixin
from classes.meta import Factory
from classes.timestep import Timestep
from updaters.coeff import BlochUpdater
from updaters.tdc import BlankTDCUpdater

fac = Factory(Molecule, MoleculeMixin)
fac.add_mixins("bloch")
mol = fac.create()(n_states = 3, n_atoms = 1, n_dim = 1, initstate = 1)

mol.bloch_n3[0,0] = 0.6
mol.bloch_n3[0,2] = 0.8
mol.bloch_n3[2,0] = 0.6
mol.bloch_n3[2,2] = 0.8
mol.nacdt_ss[0,1] = 1e-4
mol.nacdt_ss[1,2] = 1e-4
mol.nacdt_ss[1,0] = -mol.nacdt_ss[0,1]
mol.nacdt_ss[2,1] = -mol.nacdt_ss[1,2]

mols = [mol, mol.copy()]
ts = Timestep.select("const")(dt = 1)
ts.step = 10

tdcupd = BlankTDCUpdater()
cupd = BlochUpdater(n_substep = 10)
tdcupd.run(mols, ts)
cupd.run(mols, ts)
mols[-1].bloch_n3 = cupd.bloch.out

print(0.5 * np.sum(np.abs(np.abs(mols[-1].coeff_s)**2 - np.abs(mols[-2].coeff_s)**2)))
def func(mols: list[Molecule]):
    mol = mols[-1]
    tot = 0
    for i in range(mol.n_states):
        for j in range(i):
            f1 = np.abs(mols[-2].nacdt_ss[i,j])
            f2 = np.abs(mols[-1].nacdt_ss[i,j])
            tdc = max(f2, 2*f2 - f1)
            tot += np.abs(tdc) * np.sqrt(1 - (np.abs(mol.coeff_s[i])**2 - np.abs(mol.coeff_s[j])**2)**2)
    return tot
print(func(mols))
breakpoint()