import numpy as np
import os, sys, shutil
import json

from classes.molecule import Molecule

from .base import ESTProgram, est_method

from .molpro import Molpro

from pyscf import tools, gto
from functools import reduce

class Bagel(ESTProgram):
    key = "bagel"

    def initiate(self):

        fs = os.listdir(os.getcwd())

        if f'{self._file}.wf' in fs:
            os.rename(f'{self._file}.wf',f'{self._file}.wf.archive')

    def execute(self):
        err = os.system(f"{self._path}/BAGEL {self._file}.json > {self._file}.out")
        if err:
            raise InterruptedError(f"Error in BAGEL, exit code {err}")
        # self.clean_dir()

    def write(self, mol: Molecule):
        self._natoms = mol.n_atoms
        self._mol = mol
        self._method(self)


    def backup_wf(self):
        os.system(f"cp est/{self._file}.wf.archive backup/")

    def recover_wf(self):
        os.system(f"cp backup/{self._file}.wf.archive est/")

    @est_method
    def cas(self):
        geom = self._create_geom_sec()

        load_ref = self._create_load_sec()

        casscf = self._create_casscf_sec()

        force = self._create_multiple_force_sec(casscf)

        save_ref = self._create_save_sec()

        molden = self._create_molden_sec()

        total = {'bagel' : [geom, load_ref, force, save_ref, molden]}

        file = open(f'{self._file}.json','w')
        json.dump(total, file, indent = 4)

    @est_method
    def caspt2(self):
        geom = self._create_geom_sec()

        load_ref = self._create_load_sec()

        # casscf = self._create_casscf_sec()

        method = self._create_caspt2_sec()

        force = self._create_multiple_force_sec(method)

        save_ref = self._create_save_sec()

        molden = self._create_molden_sec()

        total = {'bagel' : [geom, load_ref, force, save_ref, molden]}
        # total = {'bagel' : [geom, load_ref, casscf, force, save_ref, molden]}

        file = open(f'{self._file}.json','w')
        json.dump(total, file, indent = 4)

    @est_method
    def mcqdpt2(self):
        #requires JW Park's bagel patch

        geom = self._create_geom_sec()

        load_ref = self._create_load_sec()

        # casscf = self._create_casscf_sec()

        qdpt2  = self._create_qdpt2_sec()

        force = []
        for i in range(self._nstates):
            if self._calc_grad[i]:
                force.append(self._create_single_force_sec(qdpt2,i))
            for j in range(i):
                if self._calc_nac[i,j]:
                    force.append(self._create_single_force_sec(qdpt2,i,target2=j))

        save_ref = self._create_save_sec()

        molden = self._create_molden_sec()

        # total = {'bagel' : [geom, load_ref, casscf, *force, save_ref, molden]}
        total = {'bagel' : [geom, load_ref, *force, save_ref, molden]}

        file = open(f'{self._file}.json','w')
        json.dump(total, file, indent = 4)

    @est_method
    def dsrgmrpt2(self):
        #requires JW Park's bagel patch


        geom = self._create_geom_sec()

        load_ref = self._create_load_sec()

        casscf = self._create_casscf_sec()

        dsrg = self._create_dsrg_sec()

        force = []
        for i in range(self._nstates):
            if self._calc_grad[i]:
                force.append(self._create_single_force_sec(dsrg,i))
            for j in range(i):
                if self._calc_nac[i,j]:
                    force.append(self._create_single_force_sec(dsrg,i,target2=j))

        save_ref = self._create_save_sec()

        molden = self._create_molden_sec()

        total = {'bagel' : [geom, load_ref, casscf, *force, save_ref, molden]}

        file = open(f'{self._file}.json','w')
        json.dump(total, file, indent = 4)



        #
        #
        # for i, j, val in read_output_molcas_ham(self._file+'.log', traj.est.config):
        #     if i-skip >= 0 and j-skip >= 0 and i-skip < traj.par.n_states and j-skip < traj.par.n_states:
        #         traj.pes.ham_diab_mnss[-1,0,i-skip,j-skip] = val
        #
        # for s1 in range(traj.par.n_states):
        #     traj.pes.
        #     for s2 in range(i + 1, traj.par.n_states):
        #         if traj.est.calculate_nacs[i, j]:
        #             for i, j, a, val in read_output_molcas_nac(f"molcas_{i}_{j}.log", i, j): traj.pes.nac_ddr_mnssad[-1,0,i-skip,j-skip,a] = val
        #


    def _create_geom_sec(self):
        geometry = []
        atoms = self._mol.name_a
        geom = self._mol.pos_ad
        for i, atom in enumerate(atoms):
            geometry.append({'atom' : atom.astype('<U2'), "xyz" : list(geom[i,:])})

        geom = {'title' : 'molecule'}
        geom["geometry"] = geometry

        geom['basis'] = self._options['basis']
        geom['df_basis'] = self._options['dfbasis']
        geom['angstrom'] = False
        geom['cfmm'] = False
        geom['dkh'] = False
        geom['magnetic_field'] = [0.,0.,0.]
        geom['basis_type'] = 'gaussian'
        geom['cartesian'] = False

        return geom

    def _create_load_sec(self):

        load_ref = {"title" : "load_ref"}
        load_ref['continue_geom'] = False
        load_ref['file'] = f"{self._file}.wf"

        return load_ref

    def _create_save_sec(self):

        save_ref = {"title" : "save_ref"}
        save_ref['file'] = f"{self._file}.wf"

        return save_ref

    def _create_molden_sec(self):

        molden = {"title" : "print"}
        molden["file"] = f"{self._file}.molden"
        molden["orbitals"] = True

        return molden

    def _create_multiple_force_sec(self, method):
        force = {'title' : 'forces'}
        force['export'] = True
        force['grads'] = []
        for i in range(self._options['sa']):
            try:
                if self._calc_grad[i]:
                    force['grads'].append({"title" : "force", "target" : i})
                for j in range(i+1, self._options['sa']):
                    if self._calc_nac[i,j]:
                        force['grads'].append({"title" : "nacme", "target" : i, "target2" : j, "nacmetype" : "full"})
            except IndexError:
                continue

        force['method'] = [method]
        return force

    def _create_single_force_sec(self, method, target, target2=None):
        if target2 is None:
            force = {'title' : 'force'}
        else:
            force = {'title' : 'nacme'}
            force['target2'] = target2
            force['target2'] = self._options.get('nacmtype','full')


        force['target'] = target
        force['method'] = [method]
        force['export'] = True
        return force


    def _create_casscf_sec(self):

        casscf = {'title' : 'casscf'}
        casscf['nstate'] = self._options['sa']
        nact = self._options['active']-self._options['closed']
        casscf['nact'] = nact
        casscf['nclosed'] = self._options['closed']
        casscf['charge'] = 0
        casscf['nspin'] = self._options['nel']%2
        casscf['algorithm'] = 'second'
        casscf['canonical'] = True
        casscf['fci_algorithm'] = 'kh'
        casscf['thresh'] = 1e-8
        casscf['print_thresh'] = 1e-10
        casscf['thesh_micro'] = 5e-6
        casscf['maxiter'] = 200

        return casscf

    def _create_caspt2_sec(self):


        caspt2 = {'method' : 'caspt2'}
        caspt2['sssr'] : True
        caspt2['ms'] = True
        caspt2['xms'] = True
        if self._options['ms_type'] == 'M':
            caspt2['xms'] = False
        if self._options['imag'] > 0:
            caspt2['shift_imag'] = True
            caspt2['shift'] = self._options['imag']
        else:
            caspt2['shift_imag'] = False
            caspt2['shift'] = self._options['shift']

        caspt2['frozen'] = True
        caspt2['save_ref'] = 'pt2.ref'
        caspt2['print_thresh'] = 1e-10

        method = {"title" : "caspt2"}
        method['nstate'] = self._options['sa']
        nact = self._options['active']-self._options['closed']
        method['nact'] = nact
        method['nclosed'] = self._options['closed']
        method['smith'] = caspt2



        return method

    def _create_dsrg_sec(self):

        dsrg = {"title" : "dsrgmrpt2"}

        dsrg["flow_parameter"] = self._options['flow_parameter']
        dsrg["nstate"] = self._options["sa"]
        dsrg["print_thresh"] = -1
        dsrg["fci_relax"] = self._options["fci_relax"]
        nact = self._options['active']-self._options['closed']
        dsrg['nact'] = nact
        dsrg['nclosed'] = self._options['closed']
        dsrg['maxiter'] = 200

        return dsrg

    def _create_qdpt2_sec(self):

        qdpt2 = {"title" : "mcqdpt2"}

        qdpt2["xmc"] = self._options["xmc"]
        qdpt2["shift"] = self._options["shift"]
        qdpt2["nstate"] = self._options["sa"]
        qdpt2["print_thresh"] = 1e-10
        nact = self._options['active']-self._options['closed']
        qdpt2['nact'] = nact
        qdpt2['nclosed'] = self._options['closed']
        qdpt2['maxiter'] = 200
        qdpt2['resolvent_fitting'] = self._options.get('resolvent_fitting',False)

        return qdpt2


    def _get_trans(self):
        self._trans = np.eye(self._options['sa'])
        # if self._method_name == 'caspt2':
        #
        #     NM = self._options['ms_type'] != 'M'
        #
        #     with open(f"{self._file}.out",'r') as f:
        #         for line in f:
        #             if f'* {NM*"X"}MS-CASPT2 rotation matrix' in line:
        #                 for i in range(self._options['sa']):
        #                     self._trans[i,:] = [float(q) for q in f.readline().split()]
        #                 break
        #
        if self._method_name == 'dsrgmrpt2':
            if not self._options['fci_relax']:
                with open(f"{self._file}.out",'r') as f:
                    for line in f:
                        if '++ eigenvectors ++' in line:
                            f.readline()
                            f.readline()
                            for i in range(self._options['sa']):
                                self._trans[i,:] = [float(q) for q in f.readline().split()[1:]]
            # print(self._trans)

        elif self._method_name == 'mcqdpt2':
            if self._options['xmc']:
                string = '++ Rotation Matrix from CASSCF States (XMC) ++'
            else:
                string = '++ MCQDPT2 Rotation Matrix ++'

            with open(f"{self._file}.out",'r') as f:
                for line in f:
                    if string in line:
                        f.readline()
                        f.readline()
                        for i in range(self._options['sa']):
                            self._trans[i,:] = [float(q) for q in f.readline().split()[1:]]

        print(self._trans)





    def read_ham(self):
        ham = np.diag(np.genfromtxt('ENERGY.out'))

        self._get_trans()

        return ham

    def read_grad(self):
        grad = np.zeros((self._nstates, self._natoms, 3))
        for s1 in range(self._nstates):
            if self._calc_grad[s1]:
                grad[s1] = np.genfromtxt(f'FORCE_{s1}.out',skip_header=1,usecols=(1,2,3))
        return grad


    def read_nac(self):
        nac = np.zeros((self._nstates, self._nstates, self._natoms, 3))
        for i in range(self.n_states):
            for j in range(i+1, self.n_states):
                if self._calc_nac[i,j]:
                    nac[i,j] = np.genfromtxt(f'NACME_{i}_{j}.out',skip_header=1,usecols=(1,2,3))
                    nac[j,i] = -nac[i,j]
        return nac



    def read_dipmom(self):
        dipmom = np.zeros((self._nstates, self._nstates, 3))
        with open(f"{self._file}.out", "r") as f:
            for line in f:
                if 'Expectation values' in line:
                    for d in range(3):
                        f.readline()
                        for s1 in range(self._nstates):
                            dipmom[s1,s1,d] = float(f.readline().split()[3])
                    break

            for line in f:
                if 'Transition values' in line:
                    for d in range(3):
                        f.readline()
                        for s2 in range(self._nstates):
                            for s1 in range(s2):
                                dipmom[s2,s1,d] = float(f.readline().split()[3])
                                dipmom[s1,s2,d] = dipmom[s2,s1,d]
                    break

        return dipmom

    def _write_lumorb(self, mo_coeff,lumorb_filename,nocc):
        no_ao = mo_coeff.shape[0]
        lf = open(lumorb_filename, 'w')

        lf.write(f'#INPORB 2.2\n\n\n0 1 0\n{no_ao}\n{nocc}\n#ORB\n')
        for mo in range(nocc):
            lf.write(f'* ORBITAL   1   {mo+1}\n')
            for i in range((no_ao-1)//5+1):  # first case
                lf.write(
                    ''.join([f"{elem:22.14E}" for elem in mo_coeff[5*i:5*(i+1),mo]])+'\n')

        lf.close()

    def _read_ci(self,filename):

        # we're going to try to only read the CASSCF vectors...
        # except in the case of reference-relaxed DSRG-MRPT2, where we must read the second

        ci = []

        with open(filename, 'r') as f:
            for i in range(self._options['sa']):
                # counter = 0
                ci.append({})
                for line in f:
                    if f"ci vector, state {i:3g}" in line:
                        # if self._method_name == 'mcqdpt2' and counter == 0:
                        #     counter +=1
                        #     continue
                        ci[-1] = {}
                        print(line)
                        while len(l:=f.readline().split()) == 2:
                            ci[i][self._form_key(l[0])] = float(l[1])


                        if (self._method_name != 'dsrgmrpt2' and not self._options.get('fci_relax',False)) or self._method_name == 'caspt2':
                            f.seek(0)
                            break
                f.seek(0)

        return ci

    def _write_ci(self, det, filename):

        print(det)

        keys = list(set().union(*det))

        na = keys[0].count('d') + keys[0].count('a')

        ci = np.zeros((len(keys),len(det)))

        string = f'{len(det)} {self._options["active"]} {len(keys)}\n'
        with open(filename,'w') as f:
            f.write(string)
            for ikey, key in enumerate(keys):

                sign = self._get_sign(key,na)

                for s in range(len(det)):
                    ci[ikey,s] = sign*det[s].get(key,0)

                ci[ikey] = self._trans.T @ ci[ikey]
                print(key,ci[ikey])
                f.write(key + '  '.join([f"{q:22.14e} " for q in ci[ikey]])+'\n')


    def _form_key(self, key):

        return 'd'*self._options['closed'] + key.replace('2','d').replace('.','e')

    def _get_sign(self, key, na):

        a = 0
        b = 0

        for c in key:
            if c == 'a' or c == 'd':
                a +=1
            if c == 'b' or c == 'd':
                b += na-a

        return (-1)**b







    # probably change args
    def read_ovlp(self, dummy0, dummy1, dummy2):

        if f'{self._file}.molden.old' not in os.listdir(os.getcwd()):
            shutil.copyfile(f'{self._file}.molden',f'{self._file}.molden.old')
            # shutil.copyfile(f'{self._file}.out',f'{self._file}.out.old')




        a = tools.molden.load(f'{self._file}.molden.old')
        b = tools.molden.load(f'{self._file}.molden')

        # s12 = gto.intor_cross('cint1e_ovlp_sph',a[0],b[0]).T
        mol = gto.mole.conc_mol(a[0],b[0])

        nao = a[2].shape[0]
        S = mol.intor('int1e_ovlp')
        n = np.sqrt(np.diag(S))
        S /= np.outer(n,n)
        s12 = S[:nao,nao:]
        np.savetxt('S_mix',s12,header=f"{s12.shape[0]} {s12.shape[1]}", comments='')

        # self._write_lumorb(a[2],'lumorb_a',self._options['active'])
        self._write_lumorb(b[2],'lumorb_b',self._options['active'])



        # det_a= self._read_ci(f"{self._file}.out.old")
        det_b = self._read_ci(f"{self._file}.out")


        # self._write_ci(det_a, 'dets_a')
        self._write_ci(det_b, 'dets_b')

        input_file_string = '''
    a_mo=lumorb_a
    b_mo=lumorb_b
    a_mo_read=1
    b_mo_read=1
    a_det=dets_a
    b_det=dets_b
    ao_read = 0
    '''

        with open('wf.inp', 'w') as f:
            f.write(input_file_string)

        err = os.system('$SHARC/wfoverlap.x -m 2000 -f wf.inp > wf.out')

        if err < 0:
            sys.exit()


        S_mat = Molpro._read_wf_overlap(self,'wf.out')

        shutil.copy(f'{self._file}.molden',f'{self._file}.molden.old')
        shutil.copy('dets_b','dets_a')
        shutil.copy('lumorb_b','lumorb_a')
        shutil.copy(f'{self._file}.out',f'{self._file}.out.old')

        print(S_mat)


        return S_mat



