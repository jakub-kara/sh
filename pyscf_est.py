import numpy as np
from functools import reduce
try:
    from pyscf import gto, scf, mcscf, lib, tools, ci, grad
except ImportError:
    print("Warning: pyscf module not found")
import os, sys

from errors import *
from classses.classes import Trajectory
from constants import Constants

def get_mol(traj: Trajectory, config: dict):

    geom = [[traj.geo.name_a[i], traj.geo.position_mnad[-1,0,i,:] * Constants.bohr2A] for i in range(traj.par.n_atoms)]

    mol = gto.M(atom=geom,
                basis = config['basis'],
                output = 'pyscf.out',
                verbose = 4,
                cart = False,
                spin=0,
                symmetry=False)

    return mol


def run_pyscf_cisd(traj: Trajectory, config: dict):

    mol = get_mol(traj, config)

    mf = mol.RHF().run()
    myci = mf.CISD()

    myci.nroots = config['sa']


    if traj.est.first:
        myci.kernel()
    else:
        ci0 = np.load('ci.npy')
        myci.kernel(ci0=ci0)


    np.save('ci.npy', myci.ci)

    traj.pes.ham_diab_mnss[-1,0] = np.diag(myci.e_tot)

    myci.verbose = 2
    for i in range(config['sa']):
        if traj.est.calculate_nacs[i,i]:
            traj.pes.nac_ddr_mnssad[-1,0,i,i,:,:] = myci.nuc_grad_method().kernel(myci.ci[i])


    if not traj.est.first:
        old_coeff = np.load('mo.npy')
        old_mol = lib.chkfile.load_mol('chk.chk')

        s12 = gto.intor_cross('cint1e_ovlp_sph', mol, old_mol)
        s12 = reduce(np.dot, (mf.mo_coeff.T, s12, old_coeff))
        nmo = mf.mo_energy.size
        nocc = mf.mol.nelectron // 2
        for i in range(config['sa']):
            for j in range(config['sa']):
                traj.pes.overlap_mnss[-1,0,i,j] = ci.cisd.overlap(myci.ci[i], ci0[j], nmo, nocc, s12)

    lib.chkfile.dump_mol(mol, 'chk.chk')
    np.save('mo.npy', mf.mo_coeff)








def run_pyscf_mcscf(traj: Trajectory, config: dict):

    mol = get_mol(traj, config)




    ncas = config['active'] - config['closed']
    nelecas = config['nel'] - 2*config['closed']
    nst = config['sa']

    if config['df']:
        mc = mcscf.DFCASSCF(mol, ncas, nelecas)
    else:
        mc = mcscf.CASSCF(mol, ncas, nelecas)

    mc.fix_spin_(ss=0, shift=1.0)

    mc = mc.state_average(np.ones(nst)/nst)

    mc.chkfile = 'chk.chk'

    if True :# traj.est.first:
        coeff = scf.RHF(mol).run().mo_coeff
    else:
        mc2 = lib.chkfile.load_mcscf('chk.chk')
        coeff = mcscf.project__init__guess(mc, mc2.mo_coeff)

    mc.kernel(coeff)

    tools.molden.from_mcscf(mc, 'molden.molden')

    traj.pes.ham_diab_mnss[-1,0] = np.diag(mc.e_states)
    mc_grad = mc.Gradients()
    mc_nacs = mc.nac_method()

    for i in range(nst):
        if traj.est.calculate_nacs[i,i]:
            traj.pes.nac_ddr_mnssad[-1,0,i,i,:,:] = mc_grad.kernel(state=i)
        for j in range(i):
            if traj.est.calculate_nacs[i,j]:
                traj.pes.nac_ddr_mnssad[-1,0,i,j,:,:] = mc_nacs.kernel(state=(i,j), use_etfs=True)
                traj.pes.nac_ddr_mnssad[-1,0,j,i,:,:] = -traj.pes.nac_ddr_mnssad[-1,0,i,j,:,:]
