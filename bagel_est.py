import numpy as np
import os, sys

from errors import *
from classes import Trajectory
from constants import Constants
import json


def create_input_bagel(file_root: str, config: dict, calculate_nacs: np.ndarray, atoms, geom, do_caspt2):
    """
    Creates bagel input file

    Parameters
    --------
    file_root : str
        prefix of file name (use bagel)
    config : dict
        Dictionary containing details of calculation
    calculate_nacs : np.array
        (nstate, nstate) of bools for calculation of NACMEs/gradients
    atoms : list
        list of strings of atom names
    geom : np.array
        (natoms, 3) of floats - geometry in bohr
    caspt2 : bool
        do caspt2?
    
    Modifies
    ------
    None

    Returns
    -------
    None
    """    



    print(calculate_nacs)

    geometry = []
    for i, atom in enumerate(atoms):
        geometry.append({'atom' : atom, "xyz" : list(geom[i,:])})

    geom = {'title' : 'molecule'}
    geom["geometry"] = geometry

    geom['basis'] = config['basis']
    geom['df_basis'] = config['dfbasis']
    geom['angstrom'] = False
    geom['cfmm'] = False
    geom['dkh'] = False
    geom['magnetic_field'] = [0.,0.,0.]
    geom['basis_type'] = 'gaussian'
    geom['cartesian'] = False

    load_ref = {"title" : "load_ref"}
    load_ref['continue_geom'] = False
    load_ref['file'] = "bagel.wf"

    save_ref = {"title" : "save_ref"}
    save_ref['file'] = "bagel.wf"

    casscf = {'title' : 'casscf'}
    casscf['nstate'] = config['sa']
    nact = config['active']-config['closed']
    casscf['nact'] = nact
    casscf['nclosed'] = config['closed']
    casscf['charge'] = 0
    casscf['nspin'] = config['nel']%2
    casscf['algorithm'] = 'second'
    casscf['fci_algorithm'] = 'kh'
    casscf['thresh'] = 1e-8
    casscf['thesh_micro'] = 5e-6
    casscf['maxiter'] = 200


    force = {'title' : 'forces'}
    force['export'] = True
    force['grads'] = []
    for i in range(config['sa']):
        if calculate_nacs[i,i]:
            force['grads'].append({"title" : "force", "target" : i})
        for j in range(i+1, config['sa']):
            if calculate_nacs[i,j]:
                force['grads'].append({"title" : "nacme", "target" : i, "target2" : j, "nacmetype" : "full"})




    if do_caspt2:
        caspt2 = {'method' : 'caspt2'}
        caspt2['sssr'] : True
        caspt2['ms'] = True
        caspt2['xms'] = True
        if config['ms_type'] == 'M':
            caspt2['xms'] = False
        if config['imag'] > 0:
            caspt2['shift_imag'] = True
            caspt2['shift'] = config['imag']
        else:
            caspt2['shift_imag'] = False
            caspt2['shift'] = config['shift']

        caspt2['frozen'] = True
        caspt2['sssr'] : True

        force['method'] = []
        force['method'].append({"title" : "caspt2", "smith" : caspt2, "nstate" : config['sa'], "nact" : nact, "nclosed" : config['closed']})

        total = { 'bagel' : [geom, load_ref, casscf, save_ref, force]}
    else:
        force['method'] = [casscf]
        total = {'bagel' : [geom, load_ref, force, save_ref]}

    file = open(f'{file_root}.json','w')
    json.dump(total, file, indent = 4)



def read_bagel_output(calculate_nacs, natoms, nstates):
    """
    Read bagel output file

    Parameters
    -------
    calculate_nacs : np.array
        (nstate, nstate) of bools for calculation of NACMEs/gradients
    natoms : int
        number of atoms
    nstates : int
        number of states

    Modifies
    --------
    None

    Returns
    -------
    energy : np.array
        diagonal array of energies
    nac : np.array
        (nstate,nstate,natoms,3) of floats - array of gradients/nacmes
    """
    
    energy = np.diag(np.genfromtxt('ENERGY.out'))

    nac = np.zeros((*calculate_nacs.shape, natoms, 3))
    for i in range(nstates):
        if calculate_nacs[i,i]:
            nac[i,i] = np.genfromtxt(f'FORCE_{i}.out', skip_header=1, usecols=(1,2,3))

        for j in range(i+1, nstates):
            if calculate_nacs[i,j]:
                nac[i,j] = np.genfromtxt(f'NACME_{i}_{j}.out', skip_header=1, usecols=(1,2,3))

    return energy, nac



