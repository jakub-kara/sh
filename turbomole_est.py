import sys
from errors import *
from classses.classes import Trajectory
from constants import Constants
from io_methods import write_log
import struct
import math
import numpy as np
#  from SHARC_RICC2 import *
import os



def clean_dir():
    os.system('rm control exstates')

def run_ricc2(traj: Trajectory, config):
    clean_dir()
    if True:
        os.system('cp mos old_mos')

    
    for i in range(traj.par.n_states):
        if traj.est.calculate_nacs[i,i]:
            grad_state = i
            break



    make_control_script(config, traj.geo.name_a, traj.geo.position_mnad[-1,0], grad_state)

    if True:
        os.system('cp old_mos mos')

    os.system('dscf > dscf.out')
    os.system('ricc2 > ricc2.out')
    traj.pes.ham_diab_mnss[-1,0], traj.pes.nac_ddr_mnssad[-1,0,traj.hop.active, traj.hop.active,:,:], d1 = read_ricc2_out(config)
    os.chdir('..')
    write_log(traj, f"D1 diag. = {d1}"+"\n")

    if config['sa'] > 1:
        return

    os.chdir('est')

    get_dets(config['sa'])

    move_old_files()
    os.system('cp dets dets_a')
    os.system('cp mos mos_a')

    get_ao_ovlp(config, traj.geo.name_a, traj.geo.position_mnad[-1,0], traj.geo.position_mnad[-2,0])
    

    wf_file = '''
mix_aoovl=ao_ovl
a_mo=mos_a.old
b_mo=mos_a
a_det=dets_a.old
b_det=dets_a
a_mo_read=2
b_mo_read=2
    '''
    with open('wf.inp','w') as f:
        f.write(wf_file)
    os.system('$SHARC/wfoverlap.x -f wf.inp > wf.out')

    traj.pes.overlap_mnss[-1,0] = np.eye(traj.par.n_states)
    traj.pes.overlap_mnss[-1,0,1:,1:] = read_wf_overlap('wf.out', traj.par.n_states-1)


    
def add_section(control,section):
    f = open(control, 'r')
    q = f.readlines()
    q[-1] = section+'\n'+q[-1]

    f.close()
    f = open(control, 'w')
    for l in q:
        f.write(l)
    return ''

def change_option(control, option, replace):
    tmp = []
    f = open(control, 'r')
    for l in f:
        if f'{option}' in l:
            l = replace+'\n'

        tmp.append(l)
    f.close()

    f = open(control, 'w')
    for l in tmp:
        f.write(l)

def add_option(control, section, option):
    tmp = []
    f = open(control, 'r')
    for l in f:
        if f'${section}' in l:
            l += '    '+option+'\n'

        tmp.append(l)

    f.close()
    f = open(control, 'w')
    for l in tmp:
        f.write(l)

def get_ao_ovlp(config, atoms, geom1, geom2):
    run_ao_ovlp(config, atoms, geom1, geom2)
    read_ao_ovlp()


def run_ao_ovlp(config, atoms, geom1, geom2):
    os.system('rm control')
    with open('dcoord', 'w') as f:
        f.write(f'$coord natoms=   {geom1.shape[0]*2}' + '\n')
        for i in range(len(atoms)):
            f.write(f'{geom1[i,0]}   {geom1[i,1]}  {geom1[i,2]}   {atoms[i]}'+'\n')
        for i in range(len(atoms)):
            f.write(f'{geom2[i,0]}   {geom2[i,1]}  {geom2[i,2]}   {atoms[i]}'+'\n')
        f.write('$end')

    with open('sim', 'w') as f:
        f.write('\n\n\n')
        f.write('a dcoord\nsy c1\n*\nno\n')
        f.write(f"{config['basis']}"+"\n*\n")
        f.write("*\nq\n")

    os.system('define < sim > define.out') 

    add_section('control', '$intsdebug 1 sao')

    change_option('control', 'scfiterlimit', '$scfiterlimit   0')

    os.system('dscf > odscf.out')




def make_control_script(config, atoms, geom1, grad_state):
    

    with open('coord', 'w') as f:
        f.write(f'$coord natoms=   {geom1.shape[0]}' + '\n')
        for i in range(len(atoms)):
            f.write(f'{geom1[i,0]}   {geom1[i,1]}  {geom1[i,2]}   {atoms[i]}'+'\n')
        f.write('$end')


    with open('sim', 'w') as f:
        f.write('\n\n\n')
        f.write('a coord\nsy c1\n*\nno\n')
        f.write(f"{config['basis']}"+"\n*\n")
        f.write("eht\n\n\n\n")

    #riadc2
        f.write('cc\nfreeze\n*\ncbas\n*\n')
        f.write('ricc2\nmodel adc(2)\n*\nexci\n'+f'irrep=a nexc={config["sa"]-1}'+'\n*\n')
        f.write('spectrum states=all operators=diplen,qudlen\n*')
        f.write('\n*\n*\n')


    os.system('define < sim > define.out')

    if grad_state >= 1:
        add_option('control', 'excitations', f'xgrad states=(a {grad_state} 1)')
    else:
        add_section('control', '$response')
        add_option('control', 'response', 'gradient')

    change_option('control', 'scfiterlimit', '$scfiterlimit   100')



def read_ricc2_out(config):

    def str2float(string):
        return float(string.replace('-.','-0.').replace('D','E'))


    ens = np.zeros(config['sa']-1)
    natoms = 6
    gradient = np.zeros((natoms, 3))
    gradient_found=False

    try:
        f = open('exstates', 'r') 
        for line in f:
            if '$excitation_energies_ADC(2)' in line:
                for i in range(config['sa']-1):
                    ens[i] = f.readline().split()[-1]

            #  if 'gradient' in line:
                #  gradient_found = True
                #  for i in range(natoms):
                    #  data = f.readline().split()[:3]
                    #  gradient[i,:] = [str2float(q) for q in data]
    except:
        FileNotFoundError

    with open('ricc2.out','r') as f:
        for line in f:
            if 'Final MP2 energy' in line:
                mp2_energy = float(line.split()[-2])
            if 'D1 diagnostic' in line:
                d1 = float(line.split()[-2])
                break
        if not gradient_found:
            for line in f:
                if 'cartesian gradient of the en' in line:
                    grad = []
                    f.readline()
                    f.readline()
                    f.readline()
                    for i in range(3):
                        grad.append([float(q.replace('D','E')) for q in f.readline().split()[1:]])
                        
                    for p in range(natoms//5):
                        f.readline()
                        f.readline()
                        for i in range(3):
                            grad[i] += [float(q.replace('D','E')) for q in f.readline().split()[1:]]

            gradient = np.array(grad).T
            
            

    print(gradient)
        


    ham = np.zeros((config['sa'],)*2)
    ham[0,0] = mp2_energy
    for i in range(config['sa']-1):
        ham[i+1,i+1] = ham[0,0] + ens[i]

    return ham, gradient, d1


                    

def read_ao_ovlp():
    with open('odscf.out' , 'r') as f:
        for line in f:
            if 'total number of SCF-basis' in line:
                no_basis = int(line.split()[-1])
                break
        S_mat = np.zeros((no_basis,)*2)
        for line in f:
            if 'OVERLAP(SAO)' in line:
                f.readline()
                for i in range(no_basis):
                    data = f.readline().split()
                    for j in range(i//10):
                        data += f.readline().split()
                    S_mat[i,:len(data)] = data
        with open('ao_ovl', 'w') as f:
            f.write(f"{no_basis//2}  "*2+'\n')
            for i in range(no_basis//2):
                f.write('  '.join(S_mat[no_basis//2 +i, :no_basis//2].astype(str))+'\n')


class civfl_ana:
    # this code is modified from the SHARC_RICC2 function, obtained under the terms of the GNU GPLV3.0 licence

    def __init__(self, path, imult, maxsqnorm=1.0, debug=False, filestr='CCRE0'):
        self.det_dict = {}  # dictionary with determinant strings and cicoefficient information
        self.sqcinorms = {}  # CI-norms
        self.path = path
        if imult not in [1, 3]:
            print('CCR* file readout implemented only for singlets and triplets!')
            sys.exit(106)
        self.mult = imult
        self.maxsqnorm = maxsqnorm
        self.debug = debug
        self.nmos = -1  # number of MOs
        self.nfrz = 0  # number of frozen orbs
        self.nocc = -1  # number of occupied orbs (including frozen)
        self.nvir = -1  # number of virtuals
        self.filestr = filestr
        self.read_control()
# ================================================== #

    def read_control(self):
        '''
        Reads nmos, nfrz, nvir from control file
        '''
        controlfile = 'control'
        with open(controlfile, 'r') as f:
            for line in f:
                if 'nbf(AO)' in line:
                    s = line.split('=')
                    self.nmos = int(s[-1])
                if '$closed shells' in line:
                    s = f.readline().split()[1].split('-')
                    self.nocc = int(s[-1])
                if 'implicit core' in line:
                    s = line.split()
                    self.nfrz = int(s[2])
        if self.nmos == -1:
            mosfile = os.path.join(self.path, 'mos')
            with open('mos','r') as f:
                for line in f:
                    if "eigenvalue" in line:
                        self.nmos = int(line.split()[0])
        if any([self.nmos == -1, self.nfrz == -1, self.nocc == -1]):
            print('Number of orbitals not found: nmos=%i, nfrz=%i, nocc=%i' % (self.nmos, self.nfrz, self.nocc))
            raise ValueError
        self.nvir = self.nmos - self.nocc
# ================================================== #

    def get_state_dets(self, state):
        """
        Get the transition matrix from CCR* file and add to det_dict.
        """
        if (self.mult, state) == (1, 1):
            det = self.det_string(0, self.nocc, 'de')
            self.det_dict[det] = {1: 1.}
            return
        try:
            filename = ('%s% 2i% 3i% 4i' % (self.filestr, 1, self.mult, state - (self.mult == 1))).replace(' ', '-')
            filename = os.path.join(self.path, filename)
            CCfile = open(filename, 'rb')
        except IOError:
            # if the files are not there, use the right eigenvectors
            filename = ('%s% 2i% 3i% 4i' % ('CCRE0', 1, self.mult, state - (self.mult == 1))).replace(' ', '-')
            filename = os.path.join(self.path, filename)
            CCfile = open(filename, 'rb')
        # skip 8 byte
        CCfile.read(8)
        # read method from 8 byte
        method = str(struct.unpack('8s', CCfile.read(8))[0])
        # skip 8 byte
        CCfile.read(8)
        # read number of CSFs from 4 byte
        nentry = struct.unpack('i', CCfile.read(4))[0]
        # skip 4 byte
        CCfile.read(4)
        # read 8 byte as long int
        versioncheck = struct.unpack('l', CCfile.read(8))[0]
        if versioncheck == 0:
            # skip 16 byte in Turbomole >=7.1
            CCfile.read(16)
        else:
            # skip 8 byte in Turbomole <=7.0
            CCfile.read(8)
        # checks
        if 'CCS' in method:
            print('ERROR: preoptimization vector found in file: %s' % (filename))
            sys.exit(108)
        if not nentry == self.nvir * (self.nocc - self.nfrz):
            print('ERROR: wrong number of entries found in file: %s' % (filename))
        # get data
        state_dict = {}
        nact = self.nocc - self.nfrz
        for iocc in range(nact):
            for ivirt in range(self.nvir):
                coef = struct.unpack('d', CCfile.read(8))[0]
                if self.mult == 1:
                    det = self.det_string(iocc + self.nfrz, self.nocc + ivirt, 'ab')
                    state_dict[det] = coef
                elif self.mult == 3:
                    det = self.det_string(iocc + self.nfrz, self.nocc + ivirt, 'aa')
                    state_dict[det] = coef
        # renormalize
        vnorm = 0.
        for i in state_dict:
            vnorm += state_dict[i]**2
        vnorm = math.sqrt(vnorm)
        # truncate data
        state_dict2 = {}
        norm = 0.
        for i in sorted(state_dict, key=lambda x: state_dict[x]**2, reverse=True):
            state_dict2[i] = state_dict[i] / vnorm
            norm += state_dict2[i]**2
            if norm > self.maxsqnorm:
                break
        # put into general det_dict, also adding the b->a excitation for singlets
        if self.mult == 1:
            for i in state_dict2:
                coef = state_dict2[i] / math.sqrt(2.)
                j = i.replace('a', 't').replace('b', 'a').replace('t', 'b')
                if i in self.det_dict:
                    self.det_dict[i][state] = coef
                else:
                    self.det_dict[i] = {state: coef}
                if j in self.det_dict:
                    self.det_dict[j][state] = -coef
                else:
                    self.det_dict[j] = {state: -coef}
        elif self.mult == 3:
            for i in state_dict2:
                coef = state_dict2[i]
                if i in self.det_dict:
                    self.det_dict[i][state] = coef
                else:
                    self.det_dict[i] = {state: coef}
# ================================================== #

    def det_string(self, fromorb, toorb, spin):
        if fromorb >= self.nocc or toorb < self.nocc or fromorb >= self.nmos or toorb >= self.nmos:
            print('Error generating determinant string!')
            raise ValueError
        string = 'd' * self.nocc + 'e' * (self.nmos - self.nocc)
        string = string[:fromorb] + spin[0] + string[fromorb + 1:toorb] + spin[1] + string[toorb + 1:]
        return string
# ================================================== #

    def sort_key(self, key):
        """
        For specifying the sorting order of the determinants.
        """
        return key.replace('d', '0').replace('a', '1').replace('b', '1')
# ================================================== #

    def sort_key2(self, key):
        """
        For specifying the sorting order of the determinants.
        """
        return key.replace('d', '0').replace('a', '0').replace('b', '1').replace('e', '1')
# ================================================== #



    def write_det_file(self, nstate, wname='dets', wform=' % 14.10f'):
        string = '%i %i %i\n' % (nstate-1, self.nmos, len(self.det_dict))
        for det in sorted(sorted(self.det_dict, key=self.sort_key2), key=self.sort_key):
            string += det
            for istate in range(2, nstate + 1): # 'changed so as to not print to ground state
                try:
                    string += wform % (self.det_dict[det][istate])
                except KeyError:
                    string += wform % (0.)
            string += '\n'
        with open(wname,'w') as f:
            f.write(string)


def get_dets(no_states):
    civ = civfl_ana('./', 1)
    for i in range(1,no_states):
        civ.get_state_dets(i+1)
    civ.write_det_file(no_states)


def read_wf_overlap(out_file, no_states):
    """
    reads output of the wfoverlap run
    """
    S_mat = np.zeros((no_states, no_states))
    with open(out_file,'r') as f:
        for line in f:
            #  if 'Orthonormalized overlap matrix' in line:
            if 'Overlap matrix' in line:
                f.readline()
                for i in range(no_states):
                    S_mat[i,:] = [float(j) for j in f.readline().split()[2:]]

    U, _, Vh = np.linalg.svd(S_mat)


    S_mat = U@Vh
    return S_mat

def move_old_files():
    """shifts files from previous calculation to new calculation"""
    os.system('mv dets_a dets_a.old')
    os.system('mv mos_a mos_a.old')




if __name__ == '_main_':

    config = {'sa' : 2, 'basis' : 'def2-SV(P)'}

    geom1 = np.array([[ 0.19976736514461   , -0.02270954846329 ,   -1.25137188508177 ],
   [-0.20056785892616   ,  0.02184878249690 ,    1.25112652757709 ],     
   [ 2.07874139579738   , -0.03906812045324 ,   -2.02571029640672 ],     
   [-1.34161800103192   , -0.04295645758796 ,   -2.57608195357430 ],     
   [-2.07844017603574   ,  0.04043116208051 ,    2.02785500384427 ],     
   [ 1.34211727505184   ,  0.04245418192706 ,    2.57418260364144 ]])
    geom2 = geom1
    atoms = ['C','C','H','H','H','H']
    os.system('rm control')
    make_control_script(config, atoms, geom1)

    run_dscf()
    run_ricc2()
    ham, gradient = read_ricc2_out(config)
    civ = civfl_ana('./', 1)
    
    no_states = 2
    for i in range(no_states):
        civ.get_state_dets(i+1)
    civ.write_det_file(no_states)

    move_old_files()
    os.system('cp dets dets_a')
    os.system('cp mos mos_a')

    get_ao_ovlp(config, atoms,geom1, geom2)

    wf_file = '''
mix_aoovl=ao_ovl
a_mo=mos_a.old
b_mo=mos_a
a_det=dets_a.old
b_det=dets_a
a_mo_read=2
b_mo_read=2
    '''
    with open('wf.inp','w') as f:
        f.write(wf_file)
    os.system('$SHARC/wfoverlap.x -f wf.inp > wf.out')
    

    S_mat = read_wf_overlap('wf.out', config['sa'])
    
    print(ham)
    print(gradient)
    print(S_mat)

