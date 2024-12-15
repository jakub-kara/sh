import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import scipy
import classes.constants 
from classes.molecule import Molecule
from electronic.electronic import ESTProgram


class VC(ESTProgram, key = "vcham"):
    r'''
    Collects and calculates energies, gradients, NACs and Hessians according to a Heidelburgian/Fitzrovian vibronic coupling hamiltonian

The Hamiltonian is parametrised as (latex code)
$$
H_{ij} = \epsilon^{i}\delta_{ij} + \eta^{ij}(1-\delta_{ij}) + \sum_{\alpha} \left[ \frac{m_{\alpha}}{2} \frac{\partial^2}{\partial q_\alpha^2}\delta_{ij}+ \frac{\omega_\alpha^2}{2}q_\alpha^2\delta_{ij} + \kappa^{i}_\alpha \sqrt{\omega_\alpha}q_\alpha\delta_{ij}+\lambda_\alpha^{ij}\sqrt{\omega_\alpha}q_\alpha(1-\delta_{ij})+\sum_\beta\left[\frac{\gamma^i_{\alpha\beta}}{2}\sqrt{\omega_\alpha\omega_\beta}q_\alpha q_\beta\delta_{ij}+\frac{\mu^{ij}_{\alpha\beta}}{2}\sqrt{\omega_\alpha\omega_\beta}q_\alpha q_\beta(1-\delta_{ij})\right]\right]
$$


This can be more easily calculated by factoring $\sqrt{\omega_a}$ into the position, giving a mass-frequency weighted system, labelled $Q_\alpha$
Then, we get...

$$
H_{ij} = \epsilon^{i}\delta_{ij} + \eta^{ij}(1-\delta_{ij}) + \sum_{\alpha} \left[ \frac{m_{\alpha}\omega_\alpha}{2} \frac{\partial^2}{\partial Q_\alpha^2}\delta_{ij}+ \frac{\omega_\alpha}{2}Q_\alpha^2\delta_{ij} + \kappa^{i}_\alpha Q_\alpha\delta_{ij}+\lambda_\alpha^{ij}Q_\alpha(1-\delta_{ij})+\sum_\beta\left[\frac{\gamma^i_{\alpha\beta}}{2}Q_\alpha Q_\beta\delta_{ij}+\frac{\mu^{ij}_{\alpha\beta}}{2}Q_\alpha Q_\beta(1-\delta_{ij})\right]\right]
$$

This form is much better, but adds an annoying term into the kinetic energy. In this code, there is no easy way around this, and so this module works in a slightly non-intuitive way.

First, we read in the geometries in a mass-weighted coordinate system, generally setting all masses to 1 a.u.. All of the dynamics external to this module works in this coordinate system, and we mention that these numbers can be quite large. As there are $3N_{at}-6$ of them, we store them as a $N_{at}-2$ shaped molecule (i.e. an (N-2)x3 array). This is read in and flattened, and then converted to mass-frequency weighted coordinates by multiplying by the root of the frequency. Internally, the potential is then calculated in these coordinates before being transformed on the way out, multiplying the gradient and NAC by the root of the frequency, and hessian by the outer product of roots of the frequencies.

We might also consider terms higher than quadratic, in which case they are labelled 'explicit' (following VCHAM nomenclature). These are labelled as [v,j,k,t1,t2,...], with term1 being (e.g.) 3^4, indicating that the third mode is taken to the fourth power. All terms in this are ONE-INDEXED. This correspond to a term in the Hamiltonian as $H_{jk} = v * Q_t1[0]^t_1[1] * Q_t2[0]^t_2[1]...$

We also note that a vibronic coupling model is a *diabatic* hamiltonian. The three diabatic states are defined at the Q=0 geometry. We therefore must diagonalise the Hamiltonian, and find the non-adiabatic couplings and gradients using Hellmann-Feynman theorem.

For reference

$i,j                     $ index the electronic states
$\alpha,\beta            $ index the normal mode coordinates
$\epsilon_{i}            $ are constant on-diagonal terms, representing excitation energies
$\eta_{ij}               $ are constant off-diagonal terms, generally representing spin-orbit couplings
$\omega_\alpha           $ are the harmonic frequencies, which define the coordinate system
$\kappa_\alpha^i         $ are the on-diagonal linear terms, which shift the minima of each state in each DOF
$\lambda_\alpha^{ij}     $ are the off-diagonal linear terms, which describe the coupling between states 
$\gamma_{\alpha\beta}^i  $ are on-diagonal quadratic (and bi-linear) terms. The quadratic terms modulate the frequencies of the individual states, the bi-linear terms change the potential based on two correlated states
$\mu{\alpha\beta}^{ij}   $ are off-diagonal quadratic (and bi-linear) terms, which couple the states based on two displacements

Be careful here, especially with the bi-linear terms. We use a factor of 1/2 in keeping with VCHAM, but be aware that our sum of $\beta$ is not truncated, and so we have both $\omega_1\omega_2$ and $\omega_2\omega_1$ terms

Finally, we note that we can make use of "inactive modes" physically. These amount to removing one of the normal mode displacements. To do this, set the inactive_mode[i] variable to True for the index you want to be inactive. 


    '''
    def __init__(self, **config):
        super().__init__(**config)
        self.file = self._path
        self.no_f = self._options['no_f']
        self.no_s = self._nstates
        self.labels = []
        self.initiated = False
        self.disp        = np.zeros(self.no_f)
        self.ham         = np.zeros((self.no_s,self.no_s))
        self.ddr         = np.zeros((self.no_f,self.no_s,self.no_s))
        self.hess_b      = False
        self.hess        = np.zeros((self.no_f,self.no_f,self.no_s,self.no_s))
        self.omega       = np.zeros(self.no_f)
        self.epsilon     = np.zeros(self.no_s)
        # each "_b" term is a boolean controlling whether we calculate the corresponding terms. Currently set to 1 and read in.
        self.eta         = np.zeros((self.no_s,self.no_s))
        self.eta_b       = True
        self.kappa       = np.zeros((self.no_f,self.no_s))
        self.kappa_b     = True
        self.lamb        = np.zeros((self.no_f,self.no_s,self.no_s))
        self.lamb_b      = True
        self.gamma       = np.zeros((self.no_f,self.no_f,self.no_s))
        self.gamma_b     = True
        self.mu          = np.zeros((self.no_f,self.no_f,self.no_s,self.no_s))
        self.mu_b        = True
        self.expl        = []
        self.expl_b      = True
        try: 
            self.inactive_modes = np.array(self._options['inactive_modes']).astype(bool)
        except KeyError:
            self.inactive_modes = np.zeros(self.no_f,dtype=bool)

        self.trans      = np.eye(self.no_s)
        self.hameig     = np.zeros((self.no_s,self.no_s))
        self.grad       = np.zeros((self.no_s,self.no_f))
        self.nacdr      = np.zeros((self.no_s,self.no_s,self.no_f))

    def _get_grad_nac(self):
        temp = np.einsum("si, dij, jr -> srd", self.trans.T, self.ddr, self.trans)
        self.nacdr = -temp * (1 - np.eye(self.no_s)[:, :, None])
        for s1 in range(self.no_s):
            self.grad[s1] = temp[s1,s1]
            for s2 in range(self.no_s):
                if s1 == s2:
                    self.nacdr[s1,s2] = 0.
                else:
                    self.nacdr[s1,s2] /= self.hameig[s2,s2] - self.hameig[s1,s1]

        if self.hess_b:
            self.hess = np.einsum('si,deij,jr -> srde',self.trans.T,self.hess,self.trans)

    def _diagonalise(self):
        try:
            eval, evec = np.linalg.eigh(self.ham)
        except np.linalg.LinAlgError:
            print("Error in diagonalisation!")
            print("Disp:")
            print(self.disp)
            print("Ham:")
            print(self.ham)
            raise RuntimeErrorError
            exit(55)
        self.trans = evec
        self.hameig = np.diag(eval)

    def write(self, mol: Molecule):
        # This is a little bit tricky..
        # Everything internally in the code is in standard bohr. But we want to run the internals in mass-frequency weighted coordinates Q = \sqrt{m\omega} x
        # Firstly, we assume that the mass of the mode is 1, and so we can remove it
        # Secondly, we use an internal variable called 'disp', which keeps the values of Q in a single array
        # internally, all parameters in this have to be converted to hartree
        # we then run the calculations in mass-frequency weighted coordinates to get the energy.
        # Then, we have to transform the gradient and nacme back to non-weighted coordinates (technically mass-weighted), but the mass is 1

        self.disp = 1.*mol.pos_ad.flatten()
        self.disp[self.inactive_modes] = 0.
        if not self.initiated:
            self.read_vcham_file(self.file)
        self.disp *= np.sqrt(self.omega)
        self.get_energy()
        self.grad[:,self.inactive_modes] = 0.
        self.nacdr[:,:,self.inactive_modes] = 0.
        self.grad  *= np.sqrt(self.omega)[None,:]
        self.nacdr *= np.sqrt(self.omega)[None,None,:]
        if self.hess_b:
            self.hess *= np.sqrt(self.omega[None,None,:,None] * self.omega[None,None,None,:])



    def execute(self):
        pass

    def read_ham(self):
        return self.hameig

    def read_grad(self):
        return self.grad.reshape(self.no_s,self.no_f//3,3)

    def read_nac(self):
        return self.nacdr.reshape(self.no_s,self.no_s,self.no_f//3,3)

    def read_hess(self):
        return self.hess

    def read_ovlp(self):
        raise NotImplementedError

    def get_energy(self):
        self.calculate_energy(self.disp)
        # self.ham *= np.eye(self.no_s)
        self._diagonalise()
        self._get_grad_nac()

    def calculate_energy(self,disp):

        In = np.eye(self.no_s)

        def symmetrise_H(H):
            return H + np.conj(H).T - H * In
        def symmetrise_DDR(DDR):
            return DDR + np.conj(DDR).transpose((0,2,1)) - DDR * In[None,:,:]
        def symmetrise_HESS(HESS):
            return HESS + np.conj(HESS).transpose((0,1,3,2)) - HESS * In[None,None,:,:]
        #setup
        H = np.zeros((self.no_s,self.no_s))
        DDR = np.zeros((self.no_f,self.no_s,self.no_s))

        if self.hess_b:
            HESS = np.zeros((self.no_f,self.no_f,self.no_s,self.no_s))

        # Excitation energy terms. No derivative
        H += np.diag(self.epsilon)

        # Constant off-diagonal terms (SOCs). No derivative
        if self.eta_b:
            H += self.eta

        # Base frequencies. V = 0.5 omega x^2, V' = omega x
        H += 0.5 * np.einsum('i,i',self.omega,disp**2) * In
        DDR += np.einsum('i,i->i',self.omega,disp)[:,None,None] * In[None,:,:]

        if self.hess_b:
            HESS += self.omega[:,None,None,None] * np.eye(self.no_f)[:,:,None,None] * In[None,None,:,:]
        
        # On-diagonal linear terms. V = kappa x, V' = kappa
        if self.kappa_b:
            H += np.einsum('ij,i->j',self.kappa,disp) * In
            DDR += self.kappa[:,:,None] * In[None,:,:]

        # Off-diagonal linear terms. V = lambda x, V' = lambda
        if self.lamb_b:
            H += np.einsum('ijk,i->jk',self.lamb,disp)
            DDR += self.lamb       

        # have to deal with the definitial factor of 1/2 for on-nuclear-diagonal gamma and mu
        if self.gamma_b:
            H += 0.5 * np.einsum('ijk,i,j->k ',self.gamma,disp,disp) * In
            DDR +=     np.einsum('ijk,i  ->jk',self.gamma,disp)[:,:,None] * In[None,:,:]
            if self.hess_b:
                HESS += self.gamma[:,:,:,None] * In[None,None,:,:]


        if self.mu_b:
            H += 0.5 * np.einsum('ijkl,i,j->kl ',self.mu,disp,disp)
            DDR +=     np.einsum('ijkl,i  ->jkl',self.mu,disp)
            if self.hess_b:
                HESS += self.mu 
        

        if self.expl_b:
            eham,eddr,ehess = self.calc_explicit(disp)
            H += eham
            DDR += eddr
            if self.hess_b:
                HESS += ehess

        H = symmetrise_H(H)
        DDR = symmetrise_DDR(DDR)

        self.ham = H
        self.ddr = DDR
        if self.hess_b:
            HESS = symmetrise_HESS(HESS)
            self.hess = HESS

        return H, DDR
    
    def calc_explicit(self,disp):

        H = np.zeros((self.no_s,self.no_s))
        DDR = np.zeros((self.no_f,self.no_s,self.no_s))
        HESS = np.zeros((self.no_f,self.no_f,self.no_s,self.no_s))

        for i in range(len(self.expl)):

            v1,j1,k1,a = self.expl[i]
            v = float(v1)/classes.constants.units['ev']
            j = int(j1)-1
            k = int(k1)-1

            aa = [[int(r) for r in q.split('^')] for q in a.split()]


            mult = 1

            for term in aa:
                mult *= disp[term[0]-1]**term[1] #* self.omega[term[0]-1]**(term[1]/2)

            H[j,k] += v * mult

            for term in aa:
                mult = 1
                #power rule...
                # It's the way it is because otherwise you divide by zero and everything is rubbish
                if len(aa) > 1:
                    for term2 in aa:
                        if term2 == term1: continue
                        mult *= disp[term[0]-1]**term[1] #* self.omega[term[0]-1]**(term[1]/2)
                    
                DDR[term[0]-1,j,k] += v * disp[term[0]-1]**(term[1]-1) * mult * term[1] #* self.omega[term[0]-1]**(term[1]/2)

            if self.hess_b:
                if len(aa) > 1:
                    raise NotImplementedError
                mult = 1
                term = aa[0]
                HESS[term[0]-1,term[0]-1,j,k] += v * disp[term[0]-1]**(term[1]-2) * mult * term[1] * (term[1]-1) #* self.omega[term[0]-1]**(term[1]/2)

        return H, DDR, HESS

    def read_vcham_file(self,filename):
        with open(filename,'r') as f:
            for line in f:
                if '*** Frequencies ***' in line:
                    for i in range(self.no_f):
                        tmp = f.readline().split()
                        self.labels.append(tmp[1])
                        self.omega[i] = float(tmp[-1])
        
                if '*** On-diagonal constants ***' in line:
                    f.readline()
                    for i in range(self.no_s):
                        tmp = f.readline().split()
                        self.epsilon[i] = float(tmp[-1])
        
                if '*** On-diagonal linear coupling constants (kappa) ***' in line:
                    f.readline()
                    for i in range(self.no_f):
                        tmp = f.readline().split()
                        self.kappa[i,:] = [float(tmp[k+2]) for k in range(self.no_s)]
                
                if '*** Off-diagonal linear (lambda) coupling constants ***' in line:
                    for i in range(self.no_s):
                        for j in range(i+1,self.no_s):
                            f.readline()
                            for k in range(self.no_f):
                                tmp = f.readline().split()
                                self.lamb[k,i,j] = float(tmp[-1])
                                # self.lamb[k,j,i] = lamb[k,i,j]
        
        #        if '*** On-diagonal quadratic constants (gamma) ***' in line:
        #            f.readline()
        #            for i in range(self.no_f):
        #                tmp = f.readline().split()
        #                gamma[i,i,:] = [float(tmp[i+2]) for i in range(self.no_s)]
        
                if '*** All on-diagonal bilinear (gamma) constants ***' in line:
                    for i in range(self.no_s):
                        f.readline()
                        for j in range(self.no_f//6 + 1):
                            f.readline()
                            for k in range(j*6,self.no_f):
                                tmp = f.readline().split()
                                self.gamma[k,j*6:min(k+1,6+j*6),i] = [float(l) for l in tmp[2:]]
                                self.gamma[j*6:min(k+1,6+j*6),k,i] = [float(l) for l in tmp[2:]]
                        f.readline()
        
                if '*** Off-diagonal bilinear (mu) constants ***' in line:
                    for i in range(self.no_s):
                        for i2 in range(i+1,self.no_s):
                            f.readline()
                            for j in range(self.no_f//6 + 1):
                                f.readline()
                                for k in range(j*6,self.no_f):
                                    tmp = f.readline().split()
                                    self.mu[k,j*6:min(k+1,6+j*6),i,i2] = [float(l) for l in tmp[2:]]
                                    self.mu[j*6:min(k+1,6+j*6),k,i,i2] = [float(l) for l in tmp[2:]]
                                    # self.mu[k,j*6:min(k+1,6+j*6),i2,i] = [float(l) for l in tmp[2:]]
                            f.readline()
        
                if '*** Explicitely added polynomial terms ***' in line:
                    self.expl = []
                    N = int(f.readline().split()[-1])
        
                    for i in range(N):
                        self.expl.append(f.readline().split()[1:])


        #set speeding up variables
        if np.all(self.eta == 0.):
            self.eta_b = False
        if np.all(self.kappa == 0.):
            self.kappa_b = False
        if np.all(self.lamb == 0.):
            self.lamb_b = False
        if np.all(self.gamma == 0.):
            self.gamma_b = False
        if np.all(self.mu == 0.):
            self.mu_b = False
        if len(self.expl) == 0:
            self.expl_b = False


        # definitionally, on-diagonal gamma terms are defined with a factor of a half in the potential

        self.omega       /= classes.constants.units['ev']
        self.epsilon     /= classes.constants.units['ev']
        self.eta         /= classes.constants.units['ev']
        self.kappa       /= classes.constants.units['ev']
        self.gamma       /= classes.constants.units['ev']
        self.lamb        /= classes.constants.units['ev']
        self.mu          /= classes.constants.units['ev']

        self.initiated = True

        # self.kappa       *= np.sqrt(self.omega[:,None])
        # self.lamb        *= np.sqrt(self.omega[:,None,None])
        # self.gamma       *= np.sqrt(self.omega[:,None,None]) * np.sqrt(self.omega[None,:,None])
        # self.mu          *= np.sqrt(self.omega[:,None,None,None]) * np.sqrt(self.omega[None,:,None,None])

        # print('Setting everything to zero for testing!!!')
        # self.omega[:] = 0.
        # self.omega[2] = 0.001
        # self.epsilon[0] = 0.
        # self.eta         = np.zeros((self.no_s,self.no_s))
        # self.kappa       = np.zeros((self.no_f,self.no_s))
        # self.lamb        = np.zeros((self.no_f,self.no_s,self.no_s))
        # self.gamma       = np.zeros((self.no_f,self.no_f,self.no_s))
        # self.gamma[2,2,0] = 0.001
        # self.mu          = np.zeros((self.no_f,self.no_f,self.no_s,self.no_s))
        # self.expl         = []
    
def save_pickle(VC,filename):
    with open(filename,'wb') as f:
        pickle.dump(VC,f)
        
def load_pickle(filename):
    with open(filename,'rb') as f:
        VC = pickle.load(f)
    return VC
    

    
    
    # print("frequencies")
    # print(frequencies)
    # print("constants")
    # print(constants)
    # print("kappa")
    # print(kappa)
    # print("lamb")
    # print(lamb)
    # print("gamma")
    # print(gamma)
    # print("mu")
    # print(mu)
    # print("explicit")
    # print(explicit)
    #
    
if __name__ == "__main__":
    no_f = 39
    no_s = 3
    par = VC(no_f=no_f,no_s=no_s)        


    par.read_vcham_file(sys.argv[1])

    # save_pickle(par,'save.pkl')
    
    mol = mol()
    mol.pos_ad = np.zeros(par.no_f)

    hs = []
    ds = []


    idx=2

    x = np.linspace(-11,-10.9,100)
    x = np.linspace(-20,20,300)
    for i in x:
        disp[idx] = i
        h, d = par.calculate_energy(mol)
        hs.append(h)
        ds.append(d[idx])

    hs = np.array(hs)
    ds = np.array(ds)

    test = (hs[1:]-hs[:-1])/(x[1]-x[0])
    print(test[:,0,1])
    print(ds[:,0,1])

    print()

    fig, ax = plt.subplots()

    ax.plot(x,hs[:,0,2])
    # ax.plot(x,ds[:,0,0])
    # ax.plot(x[1:],test[:,0,0])
    ax.plot(x,hs[:,0,2])
    ax.plot(x,hs[:,0,2])
    plt.show()
    
