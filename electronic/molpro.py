import numpy as np
import os, sys

from .electronic import ESTProgram
from classes.constants import multiplets, convert

class Molpro(ESTProgram):
    key = "molpro"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._options.setdefault("sa", self.n_states)
        self._options.setdefault("df", False)
        self._options.setdefault("dfbasis", False)

    def _select_method(self, key):
        methods = {
            "cas": self.cas,
        }
        return methods[key]

    def execute(self):
        err = os.system(f"molpro -W . -I . -d ./tmp_molpro -s {self._file}.inp")
        if err > 0:
            raise InterruptedError(f"Error in MOLPRO, exit code {err}")

    def backup_wf(self):
        os.system(f"cp est/{self._file}.wf backup/")
        # if self.any_ovlp():
        os.system(f"cp est/dets_a backup/")
        os.system(f"cp est/lumorb_a backup/")

    def recover_wf(self):
        os.system(f"cp backup/{self._file}.wf est/")

    def cas(self):
        """
        Writes an input file for a molpro calculation for nonadiabatic molecular dynamics.

        Parameters
        ----------
        states: np.ndarray
            number of states in each spin multiplicity to be included in the calculation
        file_root: str
            name that all the input/output files will share in their root
        config: dict
            specifics of the calculation
        nacs: nda
            which gradients/nacmes to calculate
        skip: int
            how many states should be skipped in indexing
        mld: bool
            should a molden file be created

        Returns
        -------
        None

        Modifies
        --------
        None
        """

        # check if soc elements should be calculated
        soc = self._states.shape[0] > 1

        # input file name
        file = f"{self._file}.inp"
        with open(file, "w") as f:
            # file and threshold section
            f.write(f"***\n")
            f.write(f"file,2,{self._file}.wf\n")
            f.write("memory,100,m\n")
            f.write("gprint,orbital=2,civector,angles=-1,distance=-1\n")
            f.write("gthresh,twoint=1.0d-13,energy=1.0d-10,gradient=1.0d-10,printci=0.000000009,thrprint=0\n") # TODO add thresholds

            # basis and geometry section
            f.write(f"basis={self._options['basis']}\n")
            f.write("symmetry,nosym;\n")
            f.write("angstrom;\n")
            f.write("orient,noorient;\n")
            f.write("geomtype=xyz;\n")
            f.write(f"geom={self._file}.xyz\n")

            # mcscf section - with second order if needed
            # MAX 10 CPMCSCF CALLS IN ONE MULTI
            # TODO: fix limit
            # check for density fitting flag
            if self._options["df"]:
                f.write("{df-multi," + f"df_basis={self._options['dfbasis']}," + "so;\n")
            else:
                f.write("{multi, so;\n")
            f.write("maxiter,40;\n")
            f.write(f"occ,{self._options['active']};\n")
            f.write(f"closed,{self._options['closed']};\n\n")
            f.write(f"tran,all,DM;\n")

            # wf calculation
            for s, n in enumerate(self._states):
                if n == 0: continue
                f.write(f"wf,{self._options['nel']},1,{s};\n")
                if soc:
                    # accounts for degeneracy
                    f.write(f"state,{n//(s+1)};\n")
                else:
                    # for singlets only, specify state-average
                    f.write(f"state,{self._options['sa']};\n")

            if not soc: f.write("print,orbitals;\n")

            # gradients
            if soc:
                # all gradients are needed for soc transformation
                for s, n in enumerate(self._states):
                    if n == 0: continue
                    for i in range(n//(s+1)):
                        record = 5000 + (s+1)*100 + i
                        f.write(f"CPMCSCF,GRAD,{i+1}.1,ms2={s},accu=1.0d-12,record={record}.1;\n")
            else:
                # otherwise only calculate the requested gradients
                for s, n in enumerate(self._states):
                    for i in range(n//(s+1)):
                        if self._calc_grad[i]:
                            record = 5000 + (s+1)*100 + i
                            f.write(f"CPMCSCF,GRAD,{i + 1}.1,ms2={s},accu=1.0d-12,record={record}.1;\n")

            # nacmes
            if soc:
                # all nacmes are needed for soc transformation
                for s, n in enumerate(self._states):
                    if n == 0: continue
                    for i in range(n//(s+1)):
                        for j in range(i):
                            record = 6000 + (s+1)*100 + i*(i-1)//2 + j
                            f.write(f"CPMCSCF,NACM,{j+1}.1,{i+1}.1,ms2={s},accu=1.0d-12,record={record}.1;\n")

            else:
                # otherwise only calculate requested nacmes
                for s, n in enumerate(self._states):
                    for i in range(n//(s+1)):
                        for j in range(i):
                            if self._calc_nac[i, j]:
                                record = 6000 + (s+1)*100 + i*(i-1)//2 + j
                                f.write(f"CPMCSCF,NACM,{j + 1}.1,{i + 1}.1,ms2={s},accu=1.0d-12,record={record}.1;\n")
                                record += 1
            f.write("}\n")

            # samc gradients
            # text in format "ms2 state state"; state takes skip into account
            if soc:
                for s, n in enumerate(self._states):
                    for i in range(n//(s+1)):
                        record = 5000 + (s+1)*100 + i
                        f.write(f"text,calc grad {s} {i} {i}\n")
                        f.write(f"{{FORCES;SAMC,{record}.1}};\n")
            else:
                for s, n in enumerate(self._states):
                    for i in range(n//(s+1)):
                        if self._calc_grad[i]:
                            record = 5000 + (s+1)*100 + i
                            f.write(f"text,calc grad {s} {i} {i}\n")
                            f.write(f"{{FORCES;SAMC,{record}.1}};\n")

            # samc nacmes
            if soc:
                for s, n in enumerate(self._states):
                    for i in range(n//(s+1)):
                        for j in range(i):
                            record = 6000 + (s+1)*100 + i*(i-1)//2 + j
                            f.write(f"text,calc nacm {s} {j} {i}\n")
                            f.write(f"{{FORCES;SAMC,{record}.1}};\n")
            else:
                for s, n in enumerate(self._states):
                    for i in range(n//(s+1)):
                        for j in range(i):
                            if self._calc_nac[i, j]:
                                record = 6000 + (s+1)*100 + i*(i-1)//2 + j
                                f.write(f"text,calc nacm {s} {j} {i}\n")
                                f.write(f"{{FORCES;SAMC,{record}.1}};\n")
                                record += 1

            # soc hamiltonian elements
            if soc:
                records = []
                # mrci
                for s, n in enumerate(self._states):
                    if n == 0: continue
                    f.write("{ci;\n")
                    f.write(f"wf,{self._options['nel']},1,{s};\n")
                    record = 4000 + (s+1)*100
                    records.append(record)
                    f.write(f"save,{record}.1;\n")
                    f.write(f"state,{n//(s+1)};\n")
                    f.write("noexc;}\n")

                # soc integrals
                f.write("lsint\n")
                f.write("{ci;\n")
                f.write(f"hlsmat,ls,{','.join([str(i) + '.1' for i in records])};\n")
                f.write("option,matel=1,hlstrans=1;\n")
                f.write("print,hls=2;}\n")

            # request molden
            if self._options.get("mld", False): f.write(f"put,molden, {self._file}.mld\n")
            f.write("---")

    def read_ham(self):
        """
        Reads a molpro output file and searches it for hamiltonian elements.
        Results are yielded one by one.

        Parameters
        ----------
        file_root: str
            name that all the input/output files will share in their root

        Returns
        -------
        spin1: int
            spin multiplicity of the ket
        spin2: int
            spin multiplicity of the bra
        state1: int
            total index of the ket
        state2: int
            total index of the bra
        float | complex
            value of the element

        Modifies
        --------
        None
        """

        ham = np.zeros((self._nstates, self._nstates), dtype=np.complex128)
        # read output file
        with open(f"{self._file}.out", "r") as file:
            while True:
                # read line while not eof
                line = file.readline()
                if not line:
                    break

                # make sure strings not case-sensitive
                line = line.strip().lower()

                # energy of state
                if line.startswith("!mcscf state") and "energy" in line:
                    data = line.split()
                    if len(data) == 5: spin = 0
                    else: spin = multiplets[data[-3]]
                    state = int(data[2].split(".")[0]) - 1
                    for s in range(spin+1):
                        temp = self._spinsum[spin] + state + s
                        if temp >= self._nstates:
                            break
                        ham[temp, temp] += float(data[-1])

                # soc matrix elements
                if line.startswith("symmetry of spin-orbit operator"):
                    line = file.readline().strip()
                    data = line.split()
                    s1 = int(float(data[-3]))
                    m1 = int(float(data[-1]))
                    line = file.readline().strip()
                    data = line.split()
                    s2 = int(float(data[-3]))
                    m2 = int(float(data[-1]))

                    while not "breit-pauli" in file.readline().lower(): pass
                    file.readline()
                    # read elements until line is empty
                    while (line := file.readline().strip().lower()):
                        data = line.split()
                        braket = data[2].replace(">", "").replace("<", "").split("|")
                        i = int(braket[0].split(".")[0]) - 1
                        j = int(braket[2].split(".")[0]) - 1
                        coup = complex(data[3].replace("i", "j"))

                        idx1 = self._spinsum[s1] + i*(s1+1) + s1-m1
                        idx2 = self._spinsum[s2] + j*(s2+1) + s2-m2
                        ham[idx1, idx2] += coup
                        ham[idx2, idx1] += coup.conjugate()

                        # only some elements provided, the rest has to be derived from symmetry
                        if m1 != 0 and m2 != 0:
                            idx1 = self._spinsum[s1] + i*(s1+1) + s1+m1
                            idx2 = self._spinsum[s2] + j*(s2+1) + s2+m2
                            ham[idx1, idx2] += coup
                            ham[idx2, idx1] += coup.conjugate()
                        if np.abs(m1+m2) <= 1:
                            if m1 != 0:
                                idx1 = self._spinsum[s1] + i*(s1+1) + s1+m1
                                idx2 = self._spinsum[s2] + j*(s2+1) + s2-m2
                                ham[idx1, idx2] += coup.conjugate()
                                ham[idx2, idx1] += coup
                            if m2 != 0:
                                idx1 = self._spinsum[s1] + i*(s1+1) + s1-m1
                                idx2 = self._spinsum[s2] + j*(s2+1) + s2+m2
                                ham[idx1, idx2] += coup.conjugate()
                                ham[idx2, idx1] += coup
        return ham

    def read_grad(self):
        """
        Reads a molpro output file and searches it for nonadiabatic coupling matrix elements.
        Results are yielded one by one.

        Parameters
        ----------
        file_root: str
            name that all the input/output files will share in their root

        Returns
        -------
        None

        Modifies
        --------
        None
        """

        grad = np.zeros((self._nstates, self._natoms, 3))
        # read output file
        with open(f"{self._file}.out", "r") as file:
            while True:
                # read line while not eof
                line = file.readline()
                if not line:
                    break

                # make sure strings not case-sensitive
                line = line.strip().lower()

                # read gradients
                if "*** calc grad" in line:
                    data = line.strip().split()
                    spin = int(data[-3])
                    state = int(data[-1])
                    while not file.readline().strip().lower().startswith("sa-mc gradient"): pass
                    for _ in range(3): file.readline()

                    # read atom by atom until line is empty
                    a = 0
                    while (line := file.readline().strip()):
                        data = line.strip().split()
                        # yield diagonal elements as required by the degeneracy
                        for s in range(spin+1):
                            idx = self._spinsum[spin] + state + s
                            grad[idx, a] = [float(x) for x in data[1:]]
                        a += 1
        return grad

    def read_nac(self):
        """
        Reads a molpro output file and searches it for nonadiabatic coupling matrix elements.

        Parameters
        ----------
        file_root: str
            name that all the input/output files will share in their root

        Returns
        -------
        None

        Modifies
        --------
        None
        """

        nac = np.zeros((self._nstates, self._nstates, self._natoms, 3))
        # read output file
        with open(f"{self._file}.out", "r") as file:
            while True:
                # read line while not eof
                line = file.readline()
                if not line:
                    break

                line = line.strip().lower()

                # read nacmes
                if "*** calc nacm" in line:
                    data = line.strip().split()
                    spin = int(data[-3])
                    state1 = int(data[-2])
                    state2 = int(data[-1])
                    while not file.readline().strip().lower().startswith("sa-mc nacme"): pass
                    for _ in range(3): file.readline()

                    # read atom by atom until line is empty
                    a = 0
                    while (line := file.readline().strip()):
                        data = line.strip().split()
                        for s in range(spin+1):
                            idx1 = self._spinsum[spin] + state1 + s
                            idx2 = self._spinsum[spin] + state2 + s
                            nac[idx1, idx2, a] = [float(x) for x in data[1:]]
                            nac[idx2, idx1, a] = [-float(x) for x in data[1:]]
                        a += 1
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

    def _get_ao_ovlp(self, atoms, geom1, geom2):
        """
        Wrapper function for getting the ao overlap matrix from molpro
        """

        self._run_ao_ovlp(atoms, geom1, geom2)
        self._get_s('molpro_overlap.out', 'S_mix')


    def _run_ao_ovlp(self, atoms, geom1, geom2):
        """
        Create double geometry file
        create minimal input file for molpro integration
        calculation overlap integrals
        """
        # write double_ao geometry

        comb_geom = np.vstack((convert(geom1, "au", "aa"), convert(geom2, "au", "aa") + 1e-5))

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
        f.write(f"basis={self._options['basis']}\n")
        f.write('symmetry,nosym;\norient,noorient;\ngeomtype=xyz;\n')
        f.write('geom=double_geom.xyz\n')
        f.write('GTHRESH,THROVL=-1e6,TWOINT=1.d9,PREFAC=1.d9\nGDIRECT\n\n')
        f.write('int\n\n')
        f.write('{matrop;load,s;print,s;}')

        f.close()

        err = os.system('molpro -W . -I . -d . molpro_overlap.inp')


    def _get_s(self, filename, s_filename):
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

    def _get_ci_and_orb_molpro(self, lumorb_filename, ci_filename):
        """
        read the molecular orbitals and ci vector from a molpro output, and put into format for wfoverlap
        needs to be run with
        gthresh, thrprint=0, printci=-1
        """

        # first get the molecular orbitals

        with open(f"{self._file}.out", 'r') as f:
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
            for i in range((no_ao-1)//5+1):  # first case
                lf.write(
                    ''.join(["%22.14E" % elem for elem in mo_coeff[mo][i]])+'\n')

        lf.close()

        cf = open(ci_filename, 'w')

        cf.write(f"{no_states} {nocc} {len(sd)}\n")
        for i, det in enumerate(sd):
            cf.write(no_c*'d'+det.replace('2', 'd').replace('0', 'e') +
                    '  ' + '  '.join(civs[i])+'\n')

        cf.close()


    def _read_wf_overlap(self, out_file):
        """
        reads output of the wfoverlap run
        """
        S_mat = np.zeros((self._nstates, self._nstates))
        with open(out_file,'r') as f:
            for line in f:
                #  if 'Orthonormalized overlap matrix' in line:
                if 'Overlap matrix' in line:
                    f.readline()
                    for i in range(self._nstates):
                        S_mat[i,:] = [float(j) for j in f.readline().split()[2:self._nstates+2]]

        U, _, Vh = np.linalg.svd(S_mat)


        S_mat = U @ Vh
        return S_mat

    # TODO: this is a temporary patch, requires more systematic fix
    def _move_old_files(self):
        """shifts files from previous calculation to new calculation"""
        os.system('cp ../backup/dets_a dets_a.old')
        os.system('cp ../backup/lumorb_a lumorb_a.old')

    # probably change args
    def read_ovlp(self, atoms, geom1, geom2):
        """wrapper function to run all calculations from given inputs"""
        self._move_old_files()
        self._get_ci_and_orb_molpro('lumorb_a', 'dets_a')

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

        self._get_ao_ovlp(atoms, geom1, geom2)

        # TODO add wfoverlap path...
        err = os.system('$SHARC/wfoverlap.x -m 2000 -f wf.inp > wf.out')

        if err < 0:
            sys.exit()


        os.system('rm molpro_overlap*')
        S_mat = self._read_wf_overlap('wf.out')
        return S_mat

