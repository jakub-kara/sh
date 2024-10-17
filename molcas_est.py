import numpy as np
import os, sys

from errors import *
from classses.classes import Trajectory
from constants import Constants

def create_input_molcas_main(file_root: str, config: dict, calculate_nacs: np.ndarray, skip: int):
    n_states = calculate_nacs.shape[0]
    os.system('rm A_PT2 GAMMA*')

    with open(f"{file_root}.input", "w") as file:
        file.write("&GATEWAY\n")
        file.write(f"COORD={file_root}.xyz\n")
        #  if SOC:
            #  f.write("AMFI")
        # Need to change to angstrom

        file.write("GROUP=NOSYM\n")
        file.write("RICD\n")
        file.write(f"BASIS={config['basis']}\n\n")

        file.write(">>> copy wf.wf JOBOLD\n")

        file.write("&SEWARD\n\n")
        file.write("&RASSCF\n")
        file.write("JOBIPH\nCIRESTART\n")
        nactel = config['nel'] - 2*config['closed']
        file.write(f"NACTEL={nactel} 0 0\n")
        ras2 = config['active'] - config['closed']
        file.write(f"RAS2={ras2}\n")
        file.write(f"INACTIVE={config['closed']}\n")
        for i in range(n_states):
            if calculate_nacs[i, i]:
                file.write(f"RLXROOT={i+skip+1}\n")
        file.write(f"CIROOT={n_states} {n_states} 1\n\n")

        file.write(">>> COPY JOB001 JOB002\n")
        file.write(">>> COPY molcas.JobIph JOB001\n")
        file.write(">>> COPY molcas.JobIph JOBOLD\n")
        file.write(">>> COPY molcas.JobIph wf.wf\n")

        if config['type'] == 'caspt2':
            file.write(f"&CASPT2\n")
            file.write(f"GRDT\n")
            file.write(f"imag={config['imag']}\n")
            file.write(f"shift={config['shift']}\n")
            file.write(f"ipea={config['ipea']}\n")
            file.write(f"sig2={config['sig2']}\n")
            file.write(f"threshold =  1e-8 1e-6\n")
            if config['ipea'] > 0:
                file.write('DORT\n')
            if config["ms_type"].upper()[0] == 'R':
                file.write(f"RMUL=all\n\n")
            elif config["ms_type"].upper()[0] == 'M':
                file.write(f"MULT=all\n\n")
            else:
                file.write(f"XMUL=all\n\n")
            file.write(">>> COPY molcas.JobMix JOB001\n\n")

        file.write(f"&ALASKA\n\n")

        file.write(f"&RASSI\n")

        if config.get("caspt2", False):
            file.write("EJOB\n")
        #  if SOC:
            #  f.write("SPIN\nSOCO=0\n")
        file.write("DIPR = -1\n")
        file.write("STOVERLAPS\n")


def create_input_molcas_nac(file_root: str, config: dict, calculate_nacs: np.ndarray, skip: int, nacidx1: int, nacidx2: int):
    n_states = calculate_nacs.shape[0]

    with open(f"{file_root}_{nacidx1}_{nacidx2}.input", "w") as file:
        file.write("&GATEWAY\n")
        file.write(f"COORD={file_root}.xyz\n")
        #  if SOC:
            #  f.write("AMFI")
        # Need to change to angstrom

        file.write("GROUP=NOSYM\n")
        file.write("RICD\n")
        file.write(f"BASIS={config['basis']}\n\n")
        file.write("&SEWARD\n\n")
        file.write("&RASSCF\n")
        file.write("JOBIPH\nCIRESTART\n")
        nactel = config['nel'] - 2*config['closed']
        file.write(f"NACTEL={nactel} 0 0\n")
        ras2 = config['active'] - config['closed']
        file.write(f"RAS2={ras2}\n")
        file.write(f"INACTIVE={config['closed']}\n")
        for i in range(n_states):
            if calculate_nacs[i, i]:
                file.write(f"RLXROOT={i+skip+1}\n")
        file.write(f"CIROOT={n_states} {n_states} 1\n\n")

        file.write(">>> COPY molcas.JobIph JOB001\n")

        if config['type'] == 'caspt2':
            file.write(f"&CASPT2\n")
            file.write(f"GRDT\n")
            if nacidx1 != nacidx2:
                file.write(f"NAC = {nacidx1+skip+1} {nacidx2+skip+1}\n")
            file.write("convergence = 1e-8\n")
            file.write(f"imag={config['imag']}\n")
            file.write(f"shift={config['shift']}\n")
            file.write(f"ipea={config['ipea']}\n")
            file.write(f"sig2={config['sig2']}\n")
            if config['ipea'] > 0:
                file.write('DORT\n')
            if config["ms_type"].upper()[0] == 'R':
                file.write(f"RMUL=all\n\n")
            elif config["ms_type"].upper()[0] == 'M':
                file.write(f"MULT=all\n\n")
            else:
                file.write(f"XMUL=all\n\n")
            file.write(">>> COPY molcas.JobMix JOB001\n\n")

        file.write(f"&ALASKA\n")
        if nacidx1 != nacidx2:
            file.write(f"NAC = {nacidx1+skip+1} {nacidx2+skip+1}\n")


def read_output_molcas_ham(file_name: str, config: dict):
    with open(file_name, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break

            line = line.strip().lower()

            if False: #SOC
                pass
            else:
                if config["type"] == 'caspt2':
                    if 'ms-caspt2 energies' in line:
                        #  while (line := file.readline()):
                        for i in range(config['sa']):
                            data = file.readline().strip().split()
                            yield int(data[-4])-1, int(data[-4])-1, float(data[-1])

                else:
                    if 'final state energy(ies)' in line:
                        for i in range(2): file.readline()
                        for i in range(config['sa']):
                        #  while (line := file.readline()):
                            line = file.readline()
                            data = line.strip().split()
                            yield int(data[-4])-1, int(data[-4])-1, float(data[-1])

def read_output_molcas_prop(file_name: str, config: dict):
    ovlp = np.zeros((config['sa']*2,config['sa']*2))
    with open(file_name, 'r') as file:
        for line in file:
            if 'OVERLAP MATRIX FOR THE ORIGINAL' in line:
                file.readline()
                for i in range(config['sa']*2):
                    data = file.readline().split()
                    for j in range(i//5):
                        data += file.readline().split()

                    ovlp[i,:len(data)] = [float(q) for q in data]

        ovlp =ovlp[config['sa']:,:config['sa']] 
        U,_,Vt = np.linalg.svd(ovlp)
        ovlp = U@Vt

    return ovlp


def read_output_molcas_grad(file_name: str, config: dict):
    with open(file_name, 'r') as file:
        while True:
            line = file.readline()
            if not line:
                break


            if 'RLXROOT' in line:
                state = int(line.split('=')[-1]) - 1

            if 'Molecular gradients' in line:
                for i in range(7): file.readline()
                
                a = 0
                while len(data := file.readline().split()) > 1:
                    yield state, state, a, [float(x) for x in data[1:]]
                    a+=1

def read_output_molcas_nac(file_name, i, j):
    with open(file_name, 'r') as file:
        if i != j:
            state1 = i
            state2 = j
        else:
            state1 = i
            state2 = i

        file.seek(0)
        while True:
            line = file.readline()
            if not line:
                break
            if 'Total derivative coupling' in line or 'Molecular gradients' in line:
                for i in range(7): file.readline()

                a = 0
                while len(data := file.readline().split()) > 1:
                    yield state1, state2, a, [float(x) for x in data[1:]]
                    yield state2, state1, a, [-float(x) for x in data[1:]]
                    a+=1
