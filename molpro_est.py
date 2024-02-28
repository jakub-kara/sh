import numpy as np
import os, sys

from errors import *
from classes import Trajectory
from constants import Constants

def create_input_molpro(file_root: str, config: dict, calculate_nacs: np.ndarray, skip: int, mld: bool = False):
    n_states = calculate_nacs.shape[0]

    file = f"{file_root}.inp"
    with open(file, "w") as f:
        #File and threshold section
        f.write(f"***\n")
        f.write(f"file,2,wf.wf\n")
        f.write("memory,100,m\n")
        f.write("gprint,orbital=2,civector,angles=-1,distance=-1\n")
        f.write(" gthresh,twoint=1.0d-13,energy=1.0d-10,gradient=1.0d-10,printci=0.000000009,thrprint=0\n") # TODO add thresholds
        f.write(f"punch,{file_root}.pun,new\n")

        # Basis and geometry section
        f.write(f"basis={config['basis']}\n")
        f.write("symmetry,nosym;\n")
        f.write("angstrom;\n")
        f.write("orient,noorient;\n")
        f.write("geomtype=xyz;\n")
        f.write(f"geom={file_root}.xyz\n")

        #mcscf section - with second order if needed
        # MAX 10 CPMCSCF CALLS IN ONE MULTI
        if config["df"]:
            f.write("{df-multi," + f"df_basis={config['dfbasis']}," + "so;\n")
        else:
            f.write("{multi, so;\n")
        f.write("maxiter,40;\n")
        f.write(f"occ,{config['active']};\n")
        f.write(f"closed,{config['closed']};\n")
        f.write(f"wf,{config['nel']},1,0;\n")
        f.write(f"state,{config['sa']};\n")
        f.write("print, orbitals;")

        record = 5100.1
        for i in range(n_states):
            if calculate_nacs[i, i]:
                f.write(f"CPMCSCF,GRAD,{i+skip+1}.1,accu=1.0d-12,record={record};\n")
                record += 1

        for i in range(n_states):
            for j in range(i + 1, n_states):
                if calculate_nacs[i, j]:
                    f.write(f"CPMCSCF,NACM,{i+skip+1}.1,{j+skip+1}.1,accu=1.0d-12,record={record};\n")
                    record += 1
        f.write("}\n")

        record = 5100.1
        for i in range(n_states):
            if calculate_nacs[i, i]:
                f.write(f"{{FORCES;SAMC,{record}}};\n")
                record += 1

        for i in range(n_states):
            for j in range(i + 1, n_states):
                if calculate_nacs[i, j]:
                    f.write(f"{{FORCES;SAMC,{record}}};\n")
                    record += 1

        if mld:
            f.write(f"put,molden, {file_root}.mld\n")
        f.write("---")

def read_output_molpro_ham(file_root: str):
    with open(f"{file_root}.out", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break

            line = line.strip().lower()

            if line.startswith("results for state "):
                state = int(line.split()[-1].split(".")[0]) - 1
                file.readline()
                line = file.readline().strip()
                data = line.split()
                yield state, state, float(data[-1])

            #SOC ham elements read here

def read_output_molpro_nac(file_root: str):
    with open(f"{file_root}.out", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break

            line = line.strip().lower()

            if line.startswith("sa-mc gradient for"):
                state = int(line.split()[-1].split(".")[0]) - 1
                for i in range(3): file.readline()
                
                a = 0
                while (line := file.readline().strip()):
                    data = line.strip().split()
                    yield state, state, a, [float(x) for x in data[1:]]
                    a += 1

            if line.strip().startswith("sa-mc nacme for"):
                state1 = int(line.split()[-3].split(".")[0]) - 1
                state2 = int(line.split()[-1].split(".")[0]) - 1
                for i in range(3): file.readline()
                
                a = 0
                while (line := file.readline().strip()):
                    data = line.strip().split()
                    yield state1, state2, a, [float(x) for x in data[1:]]
                    yield state2, state1, a, [-float(x) for x in data[1:]]
                    a += 1


def get_ao_ovlp(atoms, geom1, geom2, s_filename, basis):
    """
    Wrapper function for getting the ao overlap matrix from molpro
    """

    run_ao_ovlp(atoms, geom1, geom2, basis)
    get_s('molpro_overlap.out', 'S_mix')


def run_ao_ovlp(atoms, geom1, geom2, basis):
    """
    Create double geometry file
    create minimal input file for molpro integration
    calculation overlap integrals
    """
    # write double_ao geometry

    comb_geom = np.vstack((geom1*Constants.bohr2A, geom2*Constants.bohr2A + 1e-5))

    comb_atom = np.hstack((atoms, atoms))

    f = open('double_geom.xyz', 'w')
    f.write(f'{len(comb_atom)}\n\n')

    for i in range(len(comb_atom)):
        f.write(
                f'{comb_atom[i]}  {comb_geom[i,0]:10.8f}  {comb_geom[i,1]:10.8f}  {comb_geom[i,2]:10.8f}\n'
        )

    f.close()

    f = open('molpro_overlap.inp', 'w')
    f.write(
        'file, 2, wf.wf\nmemory,100,m\ngprint,orbital=2,civector,angles=-1,distance=-1\ngthresh,punchci=-1, printci=-1, thrprint=0\n'
    )
    f.write(f'basis={basis}\n')
    f.write('symmetry,nosym;\norient,noorient;\ngeomtype=xyz;\n')
    f.write('geom=double_geom.xyz\n')
    f.write('GTHRESH,THROVL=-1e6,TWOINT=1.d9,PREFAC=1.d9\nGDIRECT\n\n')
    f.write('int\n\n')
    f.write('{matrop;load,s;print,s;}')

    f.close()

    err = os.system('molpro -W . -I . -d . molpro_overlap.inp')


def get_s(filename, s_filename):
    """
    read output of molpro run from run_ao_ovlp
    put into a matrix called s_filename to be used by wfoverlap
    """

    def flatten_list(matrix):
        flat_list = []
        for row in matrix:
            flat_list += row
        return flat_list

    with open(filename, 'r') as f:
        for line in f:
            if 'NUMBER OF CONTRACTIONS:' in line:
                no_ao = int(line.split()[3])
                break

        s_mat = np.zeros((no_ao, no_ao))
        for line in f:
            if 'MATRIX S' in line:
                [f.readline() for i in range(3)]
                for ao in range(no_ao):
                    temp = []
                    for i in range((no_ao-1) // 10 + 1):
                        l = f.readline().split()
                        temp.append([float(j) for j in l])

                    s_mat[ao, :] = flatten_list(temp)
                    f.readline()

    no_ao_s = no_ao // 2

    s_mat = s_mat[no_ao_s:, :no_ao_s]

    f = open(s_filename, 'w')

    f.write(f'{no_ao_s} {no_ao_s}\n')
    for i in range(no_ao // 2):
        f.write('  '.join(['%22.14e' % j for j in s_mat[i, :]]) + '\n')

    f.close()



def get_ci_and_orb_molpro(filename, lumorb_filename, ci_filename):
    """
    read the molecular orbitals and ci vector from a molpro output, and put into format for wfoverlap
    needs to be run with 
    gthresh, thrprint=0, printci=-1
    """

    # first get the molecular orbitals

    with open('molpro.out', 'r') as f:
        for line in f:
            if 'NUMBER OF CONTRACTIONS:' in line:
                no_ao = int(line.split()[3])
            if 'Number of closed-shell orbitals:' in line:
                no_c = int(line.split()[4])
            if 'Number of active  orbitals:' in line:
                no_ac = int(line.split()[4])
            if 'Number of states:' in line:
                no_states = int(line.split()[3])
                break

        nocc = no_c + no_ac #m number of occupied total
        mo_coeff = []
        f.seek(0)
        for line in f:
            if 'NATURAL ORBITALS' in line:
                [f.readline() for i in range(4)] # read blank lines
                [f.readline() for i in range((no_ao-1)//10+2)] # read lines detailing the basis functions
                for mo in range(nocc):
                    temp = []
                    for i in range((no_ao-1) // 10 + 1):
                        if i == 0:
                            l = f.readline().split()[3:]
                        else:
                            l = f.readline().split()
                        temp.append([float(j) for j in l[:5]]) # split into two bits of five for easy writing later...
                        temp.append([float(j) for j in l[5:]])
                    f.readline() # one more line for spacing in file
                    mo_coeff.append(temp)

        # now read ci coefficients - read all as wfoverlap deals with it.
        f.seek(0)
        sd = []
        civs = []

        for line in f:
            if 'CI Coefficient' in line:
                f.readline()
                f.readline()
                while len(l := f.readline().split()) >= 1: # at end of ci vector there is an empty line
                    sd.append(l[0])
                    civs.append(l[1:])

    lf = open(lumorb_filename, 'w')

    lf.write(f'#INPORB 2.2\n\n\n0 1 0\n{no_ao}\n{nocc}\n#ORB\n')
    for mo in range(nocc):
        lf.write(f'* ORBITAL   1   {mo+1}\n')
        for i in range(no_ao//5+1):  # first case
            lf.write(
                ''.join(["%22.14E" % elem for elem in mo_coeff[mo][i]])+'\n')

    lf.close()

    cf = open(ci_filename, 'w')

    cf.write(f"{no_states} {nocc} {len(sd)}\n")
    for i, det in enumerate(sd):
        cf.write(no_c*'d'+det.replace('2', 'd').replace('0', 'e') +
                 '  ' + '  '.join(civs[i])+'\n')

    cf.close()


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
    os.system('mv lumorb_a lumorb_a.old')


def run_wfoverlap_molpro(output_filename, atoms, geom1, geom2, basis, no_states):
    """wrapper function to run all calculations from given inputs"""
    move_old_files()
    get_ci_and_orb_molpro(output_filename, 'lumorb_a', 'dets_a')

    input_file_string = '''
a_mo=lumorb_a.old
b_mo=lumorb_a
a_mo_read=1
b_mo_read=1
a_det=dets_a.old
b_det=dets_a
ao_read = 0
'''

    with open('wf.inp', 'w') as f:
        f.write(input_file_string)

    get_ao_ovlp(atoms, geom1, geom2, 'S_mix', basis)

    #TODO add wfoverlap path...
    err = os.system('$SHARC/wfoverlap.x -m 2000 -f wf.inp > wf.out')

    if err < 0:
        sys.exit()


    os.system('rm molpro_overlap*')
    S_mat = read_wf_overlap('wf.out', no_states)
    return S_mat

