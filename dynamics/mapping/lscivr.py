import numpy as np
from scipy.optimize import fsolve
from classes.meta import Factory
from classes.molecule import Molecule
from classes.out import Output
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram
from updaters.coeff import CoeffUpdater
from updaters.tdc import TDCUpdater

class LSCIVR(Dynamics):
    #Implements the linear semi-classical initial value representations

    #The key difference is in the propatation of the electronic degrees of freedom, which is performed in a single-excitation basis
    #We store the classical phase space variables of the electron in this basis in the coeff_s variable as z = x + ip
    key = "lscivr"

    def __init__(self, *, dynamics: dict, **config):
        config["nuclear"]["mixins"].append("mmst")
        super().__init__(dynamics=dynamics, **config)

        self.PE: PopulationEstimator = PopulationEstimator[dynamics.get("pop_est", "wigner")]()

    def mode(self, mol: Molecule):
        return ["g", "n", CoeffUpdater().mode, TDCUpdater().mode]

    def population(self, mol: Molecule, s: int):
        return self.PE.population(mol, s)

    def setup_x_p(self, mol: Molecule, s: int):
        self.PE.initial_pop(mol, s, self.PE.sr2(mol))

    def read_coeff(self, mol: Molecule, file = None):
        if file is None:
            return
        data = np.genfromtxt(file)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape != (mol.n_states, 2):
            raise ValueError(f"Invalid coeff input format in {file}")
        mol.x_s[:] = data[:,0]
        mol.p_s[:] = data[:,1]

    def potential_energy(self, mol: Molecule):
        # From Eq. 9 in https://doi.org/10.1063/5.0163371
        V_bar = 1/mol.n_states * np.sum(mol.ham_eig_ss)
        V = 0
        r2 = mol.r2
        for i in range(mol.n_states):
            for j in range(mol.n_states):
                if i==j: continue
                V += (r2[i] - r2[j]) * (mol.ham_eig_ss[i,i]-mol.ham_eig_ss[j,j])
        return V_bar + V * 1/mol.n_states * 1/4

    def calc_dPkindt(self, mol: Molecule):
        force = np.zeros_like(mol.acc_ad)

        force = - 1/mol.n_states * np.sum(mol.grad_sad,axis=0)
        x = mol.x_s
        p = mol.p_s
        r2 = mol.r2

        for i in range(mol.n_states):
            for j in range(mol.n_states):
                if i == j:
                    continue
                force -= 1/4 * 1/mol.n_states * (r2[i]-r2[j]) * (mol.grad_sad[i]-mol.grad_sad[j])
                force -= 1/2 * (p[i] * p[j] + x[i] * x[j]) * (mol.ham_eig_ss[j,j] - mol.ham_eig_ss[i,i]) * mol.nacdr_ssad[i,j]


        mol.dPkindt = force / mol.mass_a[:,None]

    def calc_dRdt(self, mol: Molecule):
        mol.dRdt = mol.mom_ad/mol.mass_a[:,None]

    def set_el_grads(self, mol):
        mol.dxdt_s[:] = 0.
        mol.dpdt_s[:] = 0.
        for i in range(mol.n_states):
            for j in range(mol.n_states):
                mol.dxdt_s[i] += mol.p_s[i] * 1/mol.n_states * (mol.ham_eig_ss[i,i]-mol.ham_eig_ss[j,j])
                mol.dxdt_s[i] += mol.x_s[j] * np.sum(mol.nacdr_ssad[j,i]*mol.mom_ad/mol.mass_a[:,None])

                mol.dpdt_s[i] -= mol.x_s[i] * 1/mol.n_states * (mol.ham_eig_ss[i,i]-mol.ham_eig_ss[j,j])
                mol.dpdt_s[i] += mol.p_s[j] * np.sum(mol.nacdr_ssad[j,i]*mol.mom_ad/mol.mass_a[:,None])

    def update_quantum(self, mols: list[Molecule], dt: float):
        def get_grad(coeff, ham, nac, vel):
                grad = np.zeros_like(mols[-2].coeff_s,dtype=np.complex128)
                for i in range(mols[-2].n_states):
                    for j in range(mols[-2].n_states):
                        grad[i] += np.imag(coeff[i]) * 1/mols[-2].n_states * np.sum(ham[i,i] - ham[j,j])
                        grad[i] += -1j * np.real(coeff[i]) * 1/mols[-2].n_states * np.sum(ham[i,i] - ham[j,j])
                        grad[i] += np.real(coeff[j]) * np.sum(nac[j,i] * vel)
                        grad[i] +=  1j * np.imag(coeff[j]) * np.sum(nac[j,i] * vel)
                return grad
        def RK4():
            def inter(a,b,dtsub):
                return (1-dtsub)*a + dtsub * b
            m1 = mols[-2]
            m2 = mols[-1]
            n = 20
            dtsub = dt/n
            mols[-1].coeff_s = 1*mols[-2].coeff_s
            for i in range(n):
                k1 = get_grad(mols[-1].coeff_s           ,inter(m1.ham_eig_ss,m2.ham_eig_ss,i/n)      ,inter(m1.nacdr_ssad,m2.nacdr_ssad,i/n)      ,inter(m1.vel_ad,m2.vel_ad,i/n))
                k2 = get_grad(mols[-1].coeff_s+dtsub*k1/2,inter(m1.ham_eig_ss,m2.ham_eig_ss,(i+0.5)/n),inter(m1.nacdr_ssad,m2.nacdr_ssad,(i+0.5)/n),inter(m1.vel_ad,m2.vel_ad,(i+0.5)/n))
                k3 = get_grad(mols[-1].coeff_s+dtsub*k2/2,inter(m1.ham_eig_ss,m2.ham_eig_ss,(i+0.5)/n),inter(m1.nacdr_ssad,m2.nacdr_ssad,(i+0.5)/n),inter(m1.vel_ad,m2.vel_ad,(i+0.5)/n))
                k4 = get_grad(mols[-1].coeff_s+dtsub*k3  ,inter(m1.ham_eig_ss,m2.ham_eig_ss,(i+1)/n)  ,inter(m1.nacdr_ssad,m2.nacdr_ssad,(i+1)/n)  ,inter(m1.vel_ad,m2.vel_ad,(i+1)/n))
                mols[-1].coeff_s += dtsub/6 * (k1+2*k2+2*k3+k4)

        mols[-2].coeff_s[:] = mols[-2].x_s + 1j*mols[-2].p_s
        RK4()
        mols[-1].x_s[:] = np.real(mols[-1].coeff_s)
        mols[-1].p_s[:] = np.imag(mols[-1].coeff_s)

    def calculate_acceleration(self, mol: Molecule):
        force = np.zeros_like(mol.acc_ad)

        force = - 1/mol.n_states * np.sum(mol.grad_sad,axis=0)
        x = mol.x_s
        p = mol.p_s
        r2 = mol.r2

        for i in range(mol.n_states):
            for j in range(mol.n_states):
                if i == j:
                    continue
                force -= 1/4 * 1/mol.n_states * (r2[i]-r2[j]) * (mol.grad_sad[i]-mol.grad_sad[j])
                force -= 1/2 * (p[i] * p[j] + x[i] * x[j]) * (mol.ham_eig_ss[j,j] - mol.ham_eig_ss[i,i]) * mol.nacdr_ssad[i,j]


        mol.acc_ad[:,:] = force / mol.mass_a[:,None]

    def prepare_dynamics(self, mols: list[Molecule], dt: float):
        super().prepare_dynamics(mols, dt)
        #current hack to generate correct initial distributinos if no file given
        self.setup_x_p(mols[-1], mols[-1].state)

class PopulationEstimator(metaclass = Factory):
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

class WignerPE(PopulationEstimator):
    key = "wigner"

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

class SemiclassicalPE(PopulationEstimator):
    key = "semiclassical"

    def sr2(mol: Molecule):
        return (3,1)

    def population(mol: Molecule, s: int):
        #semiclassical population estimator
        return 0.5 * mol.r2[s] - 0.5


class SpinMappingPE(PopulationEstimator):
    key = "spinmap"

    def sr2(mol:Molecule):
        return (8/3,2/3)
    def population(mol: Molecule, s: int):
        if mol.n_states != 3:
            raise NotImplementedError
        else:
            return 1/3 + 0.5*mol.r2[s] - 1/6 * np.sum(mol.r2)
        # return 1/mol.n_states

