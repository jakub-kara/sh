import numpy as np
from functools import reduce
import os

from .base import ESTProgram, est_method
from classes.constants import convert
from classes.config import Config
from classes.molecule import Molecule
try:
    from pyscf import gto, scf, mcscf, lib, tools, ci, grad
except ImportError:
    print("Warning: pyscf module not found")

@Config.Option("pyscf", "program")
class PySCF(ESTProgram):
    def __init__(self, **config):
        super().__init__(**config)

        self._ham = None
        self._grad = None
        self._nac = None
        self._ovl = None

    def _get_mol(self, mol: Molecule):
        assert mol.n_dim == 3, "PySCF only allows 3D molecules."
        names = mol.name_a.astype("U2")
        geom = [(names[i], mol.pos_ad[i,:]) for i in range(mol.n_atoms)]
        out = gto.M(atom=geom,
                    basis = self._options['basis'],
                    output = 'pyscf.out',
                    verbose = 4,
                    cart = False,
                    symmetry=False,
                    unit="Bohr")

        if (temp := self._options.get("spin")) is not None:
            out.spin = temp
        if (temp := self._options.get("nel")) is not None:
            out.nelectron = temp

        return out

    def write(self, mol):
        self._method(self, mol)

    @est_method
    def cisd(self, inp: Molecule):
        mol = self._get_mol(inp)

        mf = mol.RHF().run()
        myci = mf.CISD()
        myci.nroots = self._options['sa']


        if not os.path.exists("ci.npy"):
            myci.kernel()
        else:
            ci0 = np.load('ci.npy')
            myci.kernel(ci0=ci0)

        self._ham = self._ham_cisd(myci)
        self._grad = self._grad_cisd(myci)
        self._ovl = self._ovl_cisd(myci, mf, mol)

        np.save("ci.npy", myci.ci)
        lib.chkfile.dump_mol(mol, 'chk.chk')
        np.save('mo.npy', mf.mo_coeff)

    def _ham_cisd(self, obj):
        return np.diag(obj.e_tot)

    def _grad_cisd(self, obj):
        obj.verbose = 2
        grad = np.zeros((self._nstates, self._natoms, 3))
        for i in range(self._nstates):
            if self._calc_grad[i]:
                grad[i] = obj.nuc_grad_method().kernel(obj.ci[i])
        return grad

    def _ovl_cisd(self, obj, hf, mol):
        if os.path.exists("ci.npy"):
            ci0 = np.load("ci.npy")
            os.system("mv mo.npy mo.old.npy")
            os.system("mv chk.chk chk.old.chk")

            old_coeff = np.load('mo.old.npy')
            old_mol = lib.chkfile.load_mol('chk.old.chk')

            s12 = gto.intor_cross('cint1e_ovlp_sph', mol, old_mol)
            s12 = reduce(np.dot, (hf.mo_coeff.T, s12, old_coeff))
            nmo = hf.mo_energy.size
            nocc = hf.mol.nelectron // 2
            ovl = np.zeros((self._nstates, self._nstates))
            for i in range(self._nstates):
                for j in range(self._nstates):
                    ovl[i,j] = ci.cisd.overlap(obj.ci[i], ci0[j], nmo, nocc, s12)
            return ovl

        else:
            return np.eye(self._nstates)

    def read_ham(self):
        return self._ham

    def read_grad(self):
        return self._grad

    def read_nac(self):
        raise NotImplementedError

    def read_ovlp(self):
        return self._ovl

    def read_dipmom(self):
        raise NotImplementedError

    def backup_wf(self, target):
        os.system(f"cp * {target}")

    @est_method
    def mcscf(self, inp: Molecule):
        mol = self._get_mol(inp)

        ncas = self._options['active'] - self._options['closed']
        nelecas = self._options['nel'] - 2*self._options['closed']
        nst = self._options['sa']

        if self._options.get("df"):
            mc = mcscf.DFCASSCF(mol, ncas, nelecas)
        else:
            mc = mcscf.CASSCF(mol, ncas, nelecas)

        mc.fix_spin_(ss=0, shift=1.0)

        mc = mc.state_average(np.ones(nst)/nst)

        if not os.path.exists("ci.npy"):
            mc.kernel()
        else:
            ci0 = np.load('ci.npy')
            mc.kernel(ci0=ci0)

        if self._options["mld"]:
            tools.molden.from_mcscf(mc, 'molden.molden')

        self._ham = self._ham_mcscf(mc)
        self._grad = self._grad_mcscf(mc)
        self._nac = self._nac_mcscf(mc)

        np.save("ci.npy", mc.ci)


    def _ham_mcscf(self, obj):
        return np.diag(obj.e_states)

    def _grad_mcscf(self, obj):
        mc_grad = obj.Gradients()
        grad = np.zeros((self._nstates, self._natoms, 3))
        for i in range(self._nstates):
            if self._calc_grad[i]:
                grad[i] = mc_grad.kernel(state=i)
        return grad

    def _nac_mcscf(self, obj):
        mc_nacs = obj.nac_method()
        nacs = np.zeros((self._nstates, self._nstates, self._natoms, 3))
        for i in range(self._nstates):
            for j in range(i):
                if self._calc_nac[i,j]:
                    nacs[i,j] = mc_nacs.kernel(state=(i,j), use_etfs=True)

    @est_method
    def mp2(self, inp: Molecule):
        mol = self._get_mol(inp)

        if mol.spin == 0:
            mf = mol.RHF().run()
        else:
            mf = mol.UHF().run()
        mymp = mf.MP2().run()

        self._ham = self._ham_mp2(mymp)
        self._grad = self._grad_mp2(mymp)

    def _ham_mp2(self, obj):
        assert self.n_states == 1
        return np.array([[obj.e_tot]])

    def _grad_mp2(self, obj):
        assert self.n_states == 1
        temp = obj.Gradients().kernel()
        return temp.reshape((1, *temp.shape))
