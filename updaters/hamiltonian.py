import numpy as np
from .nuclear import NuclearUpdater
from classes.molecule import Molecule
from classes.timestep import Timestep
from dynamics.base import Dynamics
from electronic.base import ESTProgram

class HAM_base(NuclearUpdater):
    #For mapping Hamiltonians
    #Does Molecule need to be MoleculeMMST?
    # hardcoded for 3D ??

    def mol_to_y(self, mol: Molecule):
        ns = mol.n_states
        nq = 3*mol.n_atoms
        y = np.zeros((2*(ns+nq)))
        y[:ns] = mol.x_s
        y[ns:2*ns] = mol.p_s
        y[2*ns:2*ns+nq] = mol.pos_ad.flatten()
        y[2*ns+nq:] = mol.mom_ad.flatten()
        return y


    def grad(self, mol: Molecule, dyn: Dynamics):
        dyn.set_el_grads(mol)
        dyn.calc_dRdt(mol)
        dyn.calc_dPkindt(mol)
        grad = np.concatenate((mol.dxdt_s,mol.dpdt_s,mol.dRdt.flatten(),mol.dPkindt.flatten()))
        return grad

    def y_to_mol(self, mol: Molecule, y: np.array):
        ns = mol.n_states
        nq = 3*mol.n_atoms
        mol.x_s = y[:ns]
        mol.p_s = y[ns:2*ns]
        mol.pos_ad = y[2*ns:2*ns+nq].reshape((nq//3,3))
        mol.vel_ad= y[2*ns+nq:].reshape((nq//3,3)) / mol.mass_a[:,None]



class RK4_ham(HAM_base):
    key = "rk4_ham"
    substeps = 4
    b = np.array([1/6,1/3,1/3,1/6])
    c = np.array([0,1/2,1/2,1.])

    def update(self, mols: list[Molecule], ts: Timestep):
        # update position
        mol = mols[-1]
        out = self.out
        dyn = Dynamics()
        out.inter[0] = mol

        y = self.mol_to_y(out.inter[0])
        grad = self.grad(out.inter[0], dyn)
        k = np.zeros((self.substeps, len(y)))
        k[0] = 1*grad

        for i in range(1,self.substeps):
            out.inter[i] = mol.copy_all()
            self.y_to_mol(out.inter[i], y + ts.dt*k[i-1]*self.c[i])
            dyn.run_est(out.inter[i], mol, dyn.step_mode(out.inter[i]))
            grad = self.grad(out.inter[i], dyn)
            k[i] = 1*grad

        temp = mol.copy_all()
        self.y_to_mol(temp, y + np.sum(self.b[:,None] * k, axis=0))

        # calculate new acceleration
        dyn.run_est(temp, mol, dyn.step_mode(temp))
        grad = self.grad(temp, dyn)
        out.inter[:-1] = out.inter[1:]
        out.inter[-1] = temp
