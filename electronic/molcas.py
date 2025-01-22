import numpy as np
import os, sys

from .electronic import ESTProgram
from classes.constants import Constants

class Molcas(ESTProgram, key = "molcas"):
    def _select_method(self, key):
        methods = {
            "cas": self.cas,
            "pt2": self.pt2,
        }
        return methods[key]

    def execute(self):
        err = os.system(f"pymolcas -f -b1 {self._file}.input")
        if err:
            raise InterruptedError(f"Error in MOLCAS, exit code {err}")


    def backup_wf(self):
        os.system(f"cp est/{self._file}.wf backup/")

    def recover_wf(self):
        os.system(f"cp backup/{self._file}.wf est/")

    def cas(self):
        self._create_input_stem()

        for i in range(self.n_states):
            if self._calc_grad[i]:
                self._add_input_ddr(i,i)

        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                if self._calc_nac[i, j]:
                    self._add_input_ddr(i,j)

        self._add_input_rassi(False)

    def pt2(self):
        self._create_input_stem()

        self._add_input_pt2()

        for i in range(self.n_states):
            if self._calc_grad[i]:
                self._add_input_ddr(i,i)

        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                if self._calc_nac[i, j]:
                    self._add_input_ddr(i,j)

        self._add_input_rassi(True)

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


    def _create_input_stem(self):
        os.system('rm A_PT2 GAMMA*')

        with open(f"{self._file}.input", "w") as file:
            file.write("&GATEWAY\n")
            file.write(f"COORD={self._file}.xyz\n")
            #  if SOC:
                #  f.write("AMFI")
            # Need to change to angstrom

            file.write("GROUP=NOSYM\n")
            file.write("RICD\n")
            file.write(f"BASIS={self._options['basis']}\n\n")

            file.write(f">>> copy {self._file}.wf JOBOLD\n")

            file.write("&SEWARD\n\n")
            file.write("&RASSCF\n")
            file.write("JOBIPH\nCIRESTART\n")
            nactel = self._options['nel'] - 2 * self._options['closed']
            file.write(f"NACTEL={nactel} 0 0\n")
            ras2 = self._options['active'] - self._options['closed']
            file.write(f"RAS2={ras2}\n")
            file.write(f"INACTIVE={self._options['closed']}\n")
            for i in range(self.n_states):
                if self._calc_grad[i]:
                    file.write(f"RLXROOT={i+1}\n")
            file.write(f"CIROOT={self._options['sa']} {self._options['sa']} 1\n\n")

            file.write(">>> COPY JOB001 JOB002\n")
            file.write(">>> COPY molcas.JobIph JOB001\n")
            file.write(">>> COPY molcas.JobIph JOBOLD\n")
            file.write(f">>> COPY molcas.JobIph {self._file}.wf\n")

    def _add_input_ddr(self, i: int, j: int):
        with open(f"{self._file}.input", "a") as file:
            file.write(f"&ALASKA\n")
            if i==j:
                file.write(f"root = {i+1}\n\n")
            else:
                file.write(f"nac = {i+1} {j+1}\n\n")


    def _add_input_pt2(self):
        with open(f"{self._file}.input", "a") as file:
            file.write(f"&CASPT2\n")
            file.write(f"GRDT\n")
    
            file.write(f"imag={self._options['imag']}\n")
            file.write(f"shift={self._options['shift']}\n")
            file.write(f"ipea={self._options['ipea']}\n")
            file.write(f"sig2={self._options['sig2']}\n")
            file.write(f"threshold =  1e-8 1e-6\n")
            if self._options['ipea'] > 0:
                file.write('DORT\n')
            if self._options["ms_type"].upper()[0] == 'R':
                file.write(f"RMUL=all\n\n")
            elif self._options["ms_type"].upper()[0] == 'M':
                file.write(f"MULT=all\n\n")
            else:
                file.write(f"XMUL=all\n\n")
            file.write(">>> COPY molcas.JobMix JOB001\n\n")



    def _add_input_rassi(self, pt2:bool):
        with open(f"{self._file}.input", "a") as file:
            file.write(f"&RASSI\n")

            if pt2:
                file.write("EJOB\n")
            #  if SOC:
                #  f.write("SPIN\nSOCO=0\n")
            file.write("DIPR = -1\n")
            file.write("STOVERLAPS\n")


    def read_ham(self):
        ham = np.zeros((self._nstates, self._nstates), dtype=np.complex128)
        with open(self._file+'.log', 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break

                line = line.strip().lower()

                if False: #SOC
                    pass
                else:
                    if self._type == 'pt2':
                        if 'ms-caspt2 energies' in line or 'xdw-caspt2 energies' in line:
                            #  while (line := file.readline()):
                            for i in range(self._nstates):
                                data = file.readline().strip().split()
                                ham[int(data[3])-1, int(data[3])-1] = float(data[-1])

                    else:
                        if 'final state energy(ies)' in line:
                            for i in range(2): file.readline()
                            for i in range(self._nstates):
                            #  while (line := file.readline()):
                                line = file.readline()
                                data = line.strip().split()
                                ham[int(data[-4]) - 1, int(data[-4]) - 1] = float(data[-1])
        return ham

    def read_grad(self):
        grad = np.zeros((self._nstates, self._natoms, 3))
        for s1 in range(self._nstates):
            if self._calc_grad[s1]:
                grad[s1] = self._read_single_grad(s1)
        return grad

    def _read_single_grad(self, i):
        grad = np.zeros((self._natoms, 3))
        with open(self._file+'.log', 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break


                if f'Lagrangian multipliers are calculated for state no.   {i+1}' in line:
                    #maybe doesn't work for i>=10
                    while True:
                        line = file.readline()
                        if 'Molecular gradients' in line:
                            for i in range(7): file.readline()

                            a = 0
                            while len(data := file.readline().split()) > 1:
                                grad[a] = [float(x) for x in data[1:]]
                                a+=1
                            break
                    break
        return grad

    def read_nac(self):
        nac = np.zeros((self._nstates, self._nstates, self._natoms, 3))
        for i in range(self.n_states):
            for j in range(i+1, self.n_states):
                if self._calc_nac[i,j]:
                    nac[i,j] = self._read_single_nac(i, j)
                    nac[j,i] = -nac[i,j]
        return nac


    def _read_single_nac(self, i, j):
        nac = np.zeros((self._natoms, 3))
        with open(self._file+'.log', 'r') as file:
            while True:
                line = file.readline()
                if not line:
                    break


                if f'Lagrangian multipliers are calculated for states no.   {i+1}/  {j+1}' in line:
                    #maybe doesn't work for i>=10
                    while True:
                        line = file.readline()
                        if 'Total derivative coupling' in line:
                            for i in range(7): file.readline()

                            a = 0
                            while len(data := file.readline().split()) > 1:
                                nac[a] = [float(x) for x in data[1:]]
                                a+=1
                            break
                    break
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

    # probably change args
    def read_ovlp(self, dummy0, dummy1, dummy2):
        ovlp = np.zeros((self._options['sa']*2,self._options['sa']*2))
        with open(self._file+'.log', 'r') as file:
            for line in file:
                if 'OVERLAP MATRIX FOR THE ORIGINAL' in line:
                    file.readline()
                    for i in range(self._options['sa']*2):
                        data = file.readline().split()
                        for j in range(i//5):
                            data += file.readline().split()
    
                        ovlp[i,:len(data)] = [float(q) for q in data]
    
            ovlp =ovlp[self._options['sa']:,:self._options['sa']]
            ovlp =ovlp[:self._nstates,:self._nstates]
            U,_,Vt = np.linalg.svd(ovlp)
            ovlp = U@Vt
    
        return ovlp
