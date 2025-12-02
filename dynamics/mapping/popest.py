import numpy as np
from scipy.optimize import fsolve
from classes.config import Config
from classes.molecule import Molecule



@Config("pop_est", default="wigner", path=Config.DYNAMICS, valid=Config.STR)
class PopulationEstimator:
    def population(mol: Molecule, s: int):
        raise NotImplementedError

    def sr2(mol: Molecule):
        #return sampling radii squared for (occupiued, unoccupied)
        raise NotImplementedError

    def initial_pop(mol: Molecule, s: int,sr2):
        for i in range(mol.n_states):
            mol.x_s[i],mol.p_s[i] = 2*np.random.random(2)-1
            r2 = mol.r2[i]
            mol.x_s[i] /= np.sqrt(r2)
            mol.p_s[i] /= np.sqrt(r2)
            if i == s:
                mol.x_s[i] *= np.sqrt(sr2[0])
                mol.p_s[i] *= np.sqrt(sr2[0])
            else:
                mol.x_s[i] *= np.sqrt(sr2[1])
                mol.p_s[i] *= np.sqrt(sr2[1])

@Config.Option("wigner", "pop_est")
class WignerPE(PopulationEstimator):
    def sr2(mol: Molecule):
        def f(r):
            return 2**(mol.n_states+1)*(r**2-0.5)*np.exp(-r**2) * np.exp(-(mol.n_states-1)*0.5)-1
        #3 seems to guarantee the upper solution for N<100 which I've tried. I don't know if it eventually collapses
        a = fsolve(f, 3,maxfev=10000)[0]**2
        return (a,1/2)

    def population(mol: Molecule, s: int):
        #Wigner population estimator
        a = 2**(mol.n_states+1) * np.exp(-np.sum(mol.r2))
        return a * (mol.r2[s] - 0.5)

@Config.Option("semiclassical", "pop_est")
class SemiclassicalPE(PopulationEstimator):
    def sr2(mol: Molecule):
        return (3,1)

    def population(mol: Molecule, s: int):
        #semiclassical population estimator
        return 0.5 * mol.r2[s] - 0.5

@Config.Option("spinmap", "pop_est")
class SpinMappingPE(PopulationEstimator):
    def sr2(mol:Molecule):
        return (8/3,2/3)
    def population(mol: Molecule, s: int):
        if mol.n_states != 3:
            raise NotImplementedError
        else:
            return 1/3 + 0.5*mol.r2[s] - 1/6 * np.sum(mol.r2)
        # return 1/mol.n_states
