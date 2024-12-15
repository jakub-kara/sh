import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import scipy
import classes.constants 
from classes.molecule import Molecule
from electronic.electronic import ESTProgram

class VC(ESTProgram, key = "vcham"):
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
        self.omega       = np.zeros(self.no_f)
        self.epsilon     = np.zeros(self.no_s)
        self.eta         = np.zeros((self.no_s,self.no_s))
        self.kappa       = np.zeros((self.no_f,self.no_s))
        self.lamb        = np.zeros((self.no_f,self.no_s,self.no_s))
        self.gamma       = np.zeros((self.no_f,self.no_f,self.no_s))
        self.mu          = np.zeros((self.no_f,self.no_f,self.no_s,self.no_s))
        self.expl         = []
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

    def _diagonalise(self):
        try:
            eval, evec = np.linalg.eigh(self.ham)
        except np.linalg.LinAlgError:
            print("Error in diagonalisation!")
            print("Disp:")
            print(self.disp)
            print("Ham:")
            print(self.ham)
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



    def execute(self):
        pass

    def read_ham(self):
        return self.hameig

    def read_grad(self):
        return self.grad.reshape(self.no_s,self.no_f//3,3)

    def read_nac(self):
        return self.nacdr.reshape(self.no_s,self.no_s,self.no_f//3,3)

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
        #setup
        H = np.zeros((self.no_s,self.no_s))
        DDR = np.zeros((self.no_f,self.no_s,self.no_s))

        # Excitation energy terms. No derivative
        H += np.diag(self.epsilon)

        # Constant off-diagonal terms (SOCs). No derivative
        H += self.eta

        # Base frequencies. V = 0.5 omega x^2, V' = omega x
        H += 0.5 * np.einsum('i,i',self.omega,disp**2) * In
        DDR += np.einsum('i,i->i',self.omega,disp)[:,None,None] * In[None,:,:]
        
        # On-diagonal linear terms. V = kappa x, V' = kappa
        H += np.einsum('ij,i->j',self.kappa,disp) * In
        DDR += self.kappa[:,:,None] * In[None,:,:]

        # Off-diagonal linear terms. V = lambda x, V' = lambda
        H += np.einsum('ijk,i->jk',self.lamb,disp)
        DDR += self.lamb       

        # have to deal with the definitial factor of 1/2 for on-nuclear-diagonal gamma and mu
        H += 0.5 * np.einsum('ijk ,i,j->k  ',self.gamma,disp,disp) * In
        DDR +=     np.einsum('ijk ,i  ->jk ',self.gamma,disp)[:,:,None] * In[None,:,:]
        H += 0.5 * np.einsum('ijkl,i,j->kl ',self.mu   ,disp,disp)
        DDR +=     np.einsum('ijkl,i  ->jkl',self.mu   ,disp)
        
        eham,eddr = self.calc_explicit(disp)
        H += eham
        DDR += eddr

        H = symmetrise_H(H)
        DDR = symmetrise_DDR(DDR)

        self.ham = H
        self.ddr = DDR

        return H, DDR
    
    def calc_explicit(self,disp):

        H = np.zeros((self.no_s,self.no_s))
        DDR = np.zeros((self.no_f,self.no_s,self.no_s))

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
        return H, DDR

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
    
    disp = np.zeros(par.no_f)

    hs = []
    ds = []


    idx=2

    x = np.linspace(-11,-10.9,100)
    x = np.linspace(-20,20,300)
    for i in x:
        disp[idx] = i
        h, d = par.calculate_energy(disp)
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
    
