# import numpy as np
# import os, sys

# from .electronic import ESTProgram
# from classes.constants import Constants

# class Molcas(ESTProgram, key = "molcas"):
#     def _select_method(self, key):
#         methods = {
#             "cas": self.cas,
#         }
#         return methods[key]

#     def execute(self):
#         err = os.system(f"molpro -W . -I . -d ./tmp_molpro -s {self._file}.inp")
#         if err > 0:
#             raise InterruptedError(f"Error in MOLPRO, exit code {err}")

#         err = os.system(f"pymolcas -f -b1 molcas.input")
#         if err:
#             raise InterruptedError(f"Error in MOLCAS, exit code {err}")

#         for i in range(self.n_states):
#             for j in range(i + 1, self.n_states):
#                 if self._calc_nac[i, j]:
#                     err = os.system(f"pymolcas -f -b1 molcas_{i}_{j}.input")
#                     if err:
#                         raise InterruptedError(f"Error in MOLCAS, exit code {err}")

#     def backup_wf(self):
#         os.system(f"cp est/{self._file}.wf backup/")

#     def recover_wf(self):
#         os.system(f"cp backup/{self._file}.wf est/")

#     def cas(self):
#         self._create_input_main()

#         for i in range(self.n_states):
#             for j in range(i + 1, self.n_states):
#                 if self._calc_nac[i, j]:
#                     self._create_input_nac(i, j)

#         for i, j, val in read_output_molcas_ham('molcas.log', traj.est.config):
#             if i-skip >= 0 and j-skip >= 0 and i-skip < traj.par.n_states and j-skip < traj.par.n_states:
#                 traj.pes.ham_diab_mnss[-1,0,i-skip,j-skip] = val

#         for s1 in range(traj.par.n_states):
#             for s2 in range(i + 1, traj.par.n_states):
#                 if traj.est.calculate_nacs[i, j]:
#                     for i, j, a, val in read_output_molcas_nac(f"molcas_{i}_{j}.log", i, j): traj.pes.nac_ddr_mnssad[-1,0,i-skip,j-skip,a] = val

#     def _create_input_main(self):
#         os.system('rm A_PT2 GAMMA*')

#         with open(f"{self._file}.input", "w") as file:
#             file.write("&GATEWAY\n")
#             file.write(f"COORD={self._file}.xyz\n")
#             #  if SOC:
#                 #  f.write("AMFI")
#             # Need to change to angstrom

#             file.write("GROUP=NOSYM\n")
#             file.write("RICD\n")
#             file.write(f"BASIS={self._options['basis']}\n\n")

#             file.write(">>> copy wf.wf JOBOLD\n")

#             file.write("&SEWARD\n\n")
#             file.write("&RASSCF\n")
#             file.write("JOBIPH\nCIRESTART\n")
#             nactel = self._options['nel'] - 2 * self._options['closed']
#             file.write(f"NACTEL={nactel} 0 0\n")
#             ras2 = self._options['active'] - self._options['closed']
#             file.write(f"RAS2={ras2}\n")
#             file.write(f"INACTIVE={self._options['closed']}\n")
#             for i in range(self.n_states):
#                 if self._calc_grad[i]:
#                     file.write(f"RLXROOT={i+1}\n")
#             file.write(f"CIROOT={self.n_states} {self.n_states} 1\n\n")

#             file.write(">>> COPY JOB001 JOB002\n")
#             file.write(">>> COPY molcas.JobIph JOB001\n")
#             file.write(">>> COPY molcas.JobIph JOBOLD\n")
#             file.write(">>> COPY molcas.JobIph wf.wf\n")

#             if self._options['type'] == 'caspt2':
#                 file.write(f"&CASPT2\n")
#                 file.write(f"GRDT\n")
#                 file.write(f"imag={self._options['imag']}\n")
#                 file.write(f"shift={self._options['shift']}\n")
#                 file.write(f"ipea={self._options['ipea']}\n")
#                 file.write(f"sig2={self._options['sig2']}\n")
#                 file.write(f"threshold =  1e-8 1e-6\n")
#                 if self._options['ipea'] > 0:
#                     file.write('DORT\n')
#                 if self._options["ms_type"].upper()[0] == 'R':
#                     file.write(f"RMUL=all\n\n")
#                 elif self._options["ms_type"].upper()[0] == 'M':
#                     file.write(f"MULT=all\n\n")
#                 else:
#                     file.write(f"XMUL=all\n\n")
#                 file.write(">>> COPY molcas.JobMix JOB001\n\n")

#             file.write(f"&ALASKA\n\n")

#             file.write(f"&RASSI\n")

#             if self._options.get("caspt2", False):
#                 file.write("EJOB\n")
#             #  if SOC:
#                 #  f.write("SPIN\nSOCO=0\n")
#             file.write("DIPR = -1\n")
#             file.write("STOVERLAPS\n")


#     def _create_input_nac(self, nacidx1: int, nacidx2: int):
#         with open(f"{self._file}_{nacidx1}_{nacidx2}.input", "w") as file:
#             file.write("&GATEWAY\n")
#             file.write(f"COORD={self._file}.xyz\n")
#             #  if SOC:
#                 #  f.write("AMFI")
#             # Need to change to angstrom

#             file.write("GROUP=NOSYM\n")
#             file.write("RICD\n")
#             file.write(f"BASIS={self._options['basis']}\n\n")
#             file.write("&SEWARD\n\n")
#             file.write("&RASSCF\n")
#             file.write("JOBIPH\nCIRESTART\n")
#             nactel = self._options['nel'] - 2 * self._options['closed']
#             file.write(f"NACTEL={nactel} 0 0\n")
#             ras2 = self._options['active'] - self._options['closed']
#             file.write(f"RAS2={ras2}\n")
#             file.write(f"INACTIVE={self._options['closed']}\n")
#             for i in range(self.n_states):
#                 if self._calc_grad[i, i]:
#                     file.write(f"RLXROOT={i+1}\n")
#             file.write(f"CIROOT={self.n_states} {self.n_states} 1\n\n")

#             file.write(">>> COPY molcas.JobIph JOB001\n")

#             if self._options['type'] == 'caspt2':
#                 file.write(f"&CASPT2\n")
#                 file.write(f"GRDT\n")
#                 if nacidx1 != nacidx2:
#                     file.write(f"NAC = {nacidx1 + 1} {nacidx2 + 1}\n")
#                 file.write("convergence = 1e-8\n")
#                 file.write(f"imag={self._options['imag']}\n")
#                 file.write(f"shift={self._options['shift']}\n")
#                 file.write(f"ipea={self._options['ipea']}\n")
#                 file.write(f"sig2={self._options['sig2']}\n")
#                 if self._options['ipea'] > 0:
#                     file.write('DORT\n')
#                 if self._options["ms_type"].upper()[0] == 'R':
#                     file.write(f"RMUL=all\n\n")
#                 elif self._options["ms_type"].upper()[0] == 'M':
#                     file.write(f"MULT=all\n\n")
#                 else:
#                     file.write(f"XMUL=all\n\n")
#                 file.write(">>> COPY molcas.JobMix JOB001\n\n")

#             file.write(f"&ALASKA\n")
#             if nacidx1 != nacidx2:
#                 file.write(f"NAC = {nacidx1+1} {nacidx2+1}\n")

#     def read_ham(self):
#         """
#         Reads a molpro output file and searches it for hamiltonian elements.
#         Results are yielded one by one.

#         Parameters
#         ----------
#         file_root: str
#             name that all the input/output files will share in their root

#         Returns
#         -------
#         spin1: int
#             spin multiplicity of the ket
#         spin2: int
#             spin multiplicity of the bra
#         state1: int
#             total index of the ket
#         state2: int
#             total index of the bra
#         float | complex
#             value of the element

#         Modifies
#         --------
#         None
#         """

#         ham = np.zeros((self._nstates, self._nstates), dtype=np.complex128)
#         # read output file
#         with open(f"{self._file}.out", "r") as file:
#             while True:
#                 # read line while not eof
#                 line = file.readline()
#                 if not line:
#                     break

#                 # make sure strings not case-sensitive
#                 line = line.strip().lower()

#                 # energy of state
#                 if line.startswith("!mcscf state") and "energy" in line:
#                     data = line.split()
#                     if len(data) == 5: spin = 0
#                     else: spin = Constants.multiplets[data[-3]]
#                     state = int(data[2].split(".")[0]) - 1
#                     for s in range(spin+1):
#                         temp = self._spinsum[spin] + state + s
#                         if temp >= self._nstates:
#                             break
#                         ham[temp, temp] += float(data[-1])

#                 # soc matrix elements
#                 if line.startswith("symmetry of spin-orbit operator"):
#                     line = file.readline().strip()
#                     data = line.split()
#                     s1 = int(float(data[-3]))
#                     m1 = int(float(data[-1]))
#                     line = file.readline().strip()
#                     data = line.split()
#                     s2 = int(float(data[-3]))
#                     m2 = int(float(data[-1]))

#                     while not "breit-pauli" in file.readline().lower(): pass
#                     file.readline()
#                     # read elements until line is empty
#                     while (line := file.readline().strip().lower()):
#                         data = line.split()
#                         braket = data[2].replace(">", "").replace("<", "").split("|")
#                         i = int(braket[0].split(".")[0]) - 1
#                         j = int(braket[2].split(".")[0]) - 1
#                         coup = complex(data[3].replace("i", "j"))

#                         idx1 = self._spinsum[s1] + i*(s1+1) + s1-m1
#                         idx2 = self._spinsum[s2] + j*(s2+1) + s2-m2
#                         ham[idx1, idx2] += coup
#                         ham[idx2, idx1] += coup.conjugate()

#                         # only some elements provided, the rest has to be derived from symmetry
#                         if m1 != 0 and m2 != 0:
#                             idx1 = self._spinsum[s1] + i*(s1+1) + s1+m1
#                             idx2 = self._spinsum[s2] + j*(s2+1) + s2+m2
#                             ham[idx1, idx2] += coup
#                             ham[idx2, idx1] += coup.conjugate()
#                         if np.abs(m1+m2) <= 1:
#                             if m1 != 0:
#                                 idx1 = self._spinsum[s1] + i*(s1+1) + s1+m1
#                                 idx2 = self._spinsum[s2] + j*(s2+1) + s2-m2
#                                 ham[idx1, idx2] += coup.conjugate()
#                                 ham[idx2, idx1] += coup
#                             if m2 != 0:
#                                 idx1 = self._spinsum[s1] + i*(s1+1) + s1-m1
#                                 idx2 = self._spinsum[s2] + j*(s2+1) + s2+m2
#                                 ham[idx1, idx2] += coup.conjugate()
#                                 ham[idx2, idx1] += coup
#         return ham

#     def read_grad(self):
#         """
#         Reads a molpro output file and searches it for nonadiabatic coupling matrix elements.
#         Results are yielded one by one.

#         Parameters
#         ----------
#         file_root: str
#             name that all the input/output files will share in their root

#         Returns
#         -------
#         None

#         Modifies
#         --------
#         None
#         """

#         grad = np.zeros((self._nstates, self._natoms, 3))
#         # read output file
#         with open(f"{self._file}.out", "r") as file:
#             while True:
#                 # read line while not eof
#                 line = file.readline()
#                 if not line:
#                     break

#                 # make sure strings not case-sensitive
#                 line = line.strip().lower()

#                 # read gradients
#                 if "*** calc grad" in line:
#                     data = line.strip().split()
#                     spin = int(data[-3])
#                     state = int(data[-1])
#                     while not file.readline().strip().lower().startswith("sa-mc gradient"): pass
#                     for _ in range(3): file.readline()

#                     # read atom by atom until line is empty
#                     a = 0
#                     while (line := file.readline().strip()):
#                         data = line.strip().split()
#                         # yield diagonal elements as required by the degeneracy
#                         for s in range(spin+1):
#                             idx = self._spinsum[spin] + state + s
#                             grad[idx, a] = [float(x) for x in data[1:]]
#                         a += 1
#         return grad

#     def read_nac(self):
#         nac = np.zeros((self._nstates, self._nstates, self._natoms, 3))
#         for i in range(self.n_states):
#             for j in range(i+1, self.n_states):
#                 nac[i,j] = self._read_single_nac(i, j)
#                 nac[j,i] = -nac[i,j]



#     def _read_single_nac(self, i, j):
#         nac = np.zeros((self._natoms, 3))
#         with open(self._file, 'r') as file:
#             while True:
#                 line = file.readline()
#                 if not line:
#                     break

#                 if 'Total derivative coupling' in line or 'Molecular gradients' in line:
#                     for i in range(7): file.readline()

#                     a = 0
#                     while len(data := file.readline().split()) > 1:
#                         nac[a] = [float(x) for x in data[1:]]
#                         a += 1
#         return nac

#     def read_dipmom(self):
#         dipmom = np.zeros((self._nstates, self._nstates, 3))
#         with open(f"{self._file}.out", "r") as f:
#             for line in f:
#                 if 'Expectation values' in line:
#                     for d in range(3):
#                         f.readline()
#                         for s1 in range(self._nstates):
#                             dipmom[s1,s1,d] = float(f.readline().split()[3])
#                     break

#             for line in f:
#                 if 'Transition values' in line:
#                     for d in range(3):
#                         f.readline()
#                         for s2 in range(self._nstates):
#                             for s1 in range(s2):
#                                 dipmom[s2,s1,d] = float(f.readline().split()[3])
#                                 dipmom[s1,s2,d] = dipmom[s2,s1,d]
#                     break

#         return dipmom

#     # probably change args
#     def read_ovlp(self, atoms, geom1, geom2):
#         """wrapper function to run all calculations from given inputs"""
#         self._move_old_files()
#         self._get_ci_and_orb_molpro('lumorb_a', 'dets_a')

#         input_file_string = '''
#     a_mo=lumorb_a.old
#     b_mo=lumorb_a
#     a_mo_read=1
#     b_mo_read=1
#     a_det=dets_a.old
#     b_det=dets_a
#     ao_read = 0
#     '''

#         with open('wf.inp', 'w') as f:
#             f.write(input_file_string)

#         self._get_ao_ovlp(atoms, geom1, geom2)

#         # TODO add wfoverlap path...
#         err = os.system('$SHARC/wfoverlap.x -m 2000 -f wf.inp > wf.out')

#         if err < 0:
#             sys.exit()


#         os.system('rm molpro_overlap*')
#         S_mat = self._read_wf_overlap('wf.out')
#         return S_mat

# def read_output_molcas_ham(file_name: str, config: dict):
#     with open(file_name, 'r') as file:
#         while True:
#             line = file.readline()
#             if not line:
#                 break

#             line = line.strip().lower()

#             if False: #SOC
#                 pass
#             else:
#                 if config["type"] == 'caspt2':
#                     if 'ms-caspt2 energies' in line:
#                         #  while (line := file.readline()):
#                         for i in range(config['sa']):
#                             data = file.readline().strip().split()
#                             yield int(data[-4])-1, int(data[-4])-1, float(data[-1])

#                 else:
#                     if 'final state energy(ies)' in line:
#                         for i in range(2): file.readline()
#                         for i in range(config['sa']):
#                         #  while (line := file.readline()):
#                             line = file.readline()
#                             data = line.strip().split()
#                             yield int(data[-4])-1, int(data[-4])-1, float(data[-1])

# def read_output_molcas_prop(file_name: str, config: dict):
#     ovlp = np.zeros((config['sa']*2,config['sa']*2))
#     with open(file_name, 'r') as file:
#         for line in file:
#             if 'OVERLAP MATRIX FOR THE ORIGINAL' in line:
#                 file.readline()
#                 for i in range(config['sa']*2):
#                     data = file.readline().split()
#                     for j in range(i//5):
#                         data += file.readline().split()

#                     ovlp[i,:len(data)] = [float(q) for q in data]

#         ovlp =ovlp[config['sa']:,:config['sa']]
#         U,_,Vt = np.linalg.svd(ovlp)
#         ovlp = U@Vt

#     return ovlp


# def read_output_molcas_grad(file_name: str, config: dict):
#     with open(file_name, 'r') as file:
#         while True:
#             line = file.readline()
#             if not line:
#                 break


#             if 'RLXROOT' in line:
#                 state = int(line.split('=')[-1]) - 1

#             if 'Molecular gradients' in line:
#                 for i in range(7): file.readline()

#                 a = 0
#                 while len(data := file.readline().split()) > 1:
#                     yield state, state, a, [float(x) for x in data[1:]]
#                     a+=1

# def read_output_molcas_nac(file_name, i, j):
#     with open(file_name, 'r') as file:
#         if i != j:
#             state1 = i
#             state2 = j
#         else:
#             state1 = i
#             state2 = i

#         file.seek(0)
#         while True:
#             line = file.readline()
#             if not line:
#                 break
#             if 'Total derivative coupling' in line or 'Molecular gradients' in line:
#                 for i in range(7): file.readline()

#                 a = 0
#                 while len(data := file.readline().split()) > 1:
#                     yield state1, state2, a, [float(x) for x in data[1:]]
#                     yield state2, state1, a, [-float(x) for x in data[1:]]
#                     a+=1
