import numpy as np
import os, sys

from .base import ESTProgram, est_method
from classes.constants import multiplets, convert

class MCCI(ESTProgram):
    key = "mcci"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._options.setdefault("ieig", 3)
        self._options.setdefault("maxtry", 3000)
        self._options.setdefault("cmin", 0.0005)

    def write(self, mol):
        super().write(mol)
        with open("InputGeom.txt", "w") as f:
            for i in range(mol.n_atoms):
                for j in range(mol.n_dim):
                    f.write(f" {mol.pos_ad[i,j]}")
                f.write("\n")

    def execute(self):
        err = os.system(f"python3 {self._path}/mcci.py")
        if err > 0:
            raise InterruptedError(f"Error in MCCI, exit code {err}")

    def backup_wf(self):
        pass

    def recover_wf(self):
        pass

    @est_method
    def mcci(self):
        with open("mcciCalc.in", "w") as f:
            f.write(f"""
! mcci.in file for version four of mcci
restart         = .false.               ! .TRUE. sets inflg = 1, .FALSE sets inflg = 0
n_up            = {len(self._options["mo_up"])}                     ! number of up electrons
mo_up           = {','.join([str(x) for x in self._options["mo_up"]])} ! occupied spin up orbital labels
n_dn            = {len(self._options["mo_dn"])}
mo_dn           = {','.join([str(x) for x in self._options["mo_dn"]])} ! occupied spin up orbital labels
s               =            0.0                     ! spin ( units of h_bar ) (principle case m_s = s)
ieig            = {self._options["ieig"]}                       ! ith eigenvalue
frac            = 1.0
maxtry          = {self._options["maxtry"]}          ! how many diagonalisations?
cmin            = {self._options["cmin"]}
frozen_doubly   =  1
npfull          = 10
i_want_conv     = .true.
npfull_conv     = .true.
conv_thresh_e   = 0.0005
conv_thresh_l   = 100.0
conv_average    = 3            ! 3 fullprune steps are being averaged
conv_history    = 3            ! 3 averaged Dvalues are being tracked
SCF_integral_filename= "FCIDUMP"
run_pt2=.false.
use_sds=.false.
spin_print=.false.
sa_mcci         =.true.
!parameters
maxc = 200000
kmax = 35
maxh = 400000000
""")


        with open("mcciTRDM.in", "w") as f:
            f.write(f"""
! mcci.in file for version four of mcci
restart         = .true.               ! .TRUE. sets inflg = 1, .FALSE sets inflg = 0
n_up            = {len(self._options["mo_up"])}                     ! number of up electrons
mo_up           = {','.join([str(x) for x in self._options["mo_up"]])} ! occupied spin up orbital labels
n_dn            = {len(self._options["mo_dn"])}
mo_dn           = {','.join([str(x) for x in self._options["mo_dn"]])} ! occupied spin up orbital labels
s               =            0.0                     ! spin ( units of h_bar ) (principle case m_s = s)
ieig            = {self._options["ieig"]}                       ! ith eigenvalue
frac            = 1.0
maxtry          = 0          ! how many diagonalisations?
cmin            = {self._options["cmin"]}
frozen_doubly   =  1
npfull          = 10
i_want_conv     = .true.
npfull_conv     = .true.
conv_thresh_e   = 0.0005
conv_thresh_l   = 100.0
conv_average    = 3            ! 3 fullprune steps are being averaged
conv_history    = 3            ! 3 averaged Dvalues are being tracked
SCF_integral_filename= "FCIDUMP"
run_pt2=.false.
use_sds=.true.
spin_print=.false.
sa_mcci         =.true.
!parameters
maxc = 200000
kmax = 35
maxh = 400000000
""")



    def read_ham(self):
        en = np.genfromtxt("FinalE")
        return np.diag(en[:self.n_states])

    def read_grad(self):
        grad = np.full((self._nstates, self._natoms, 3), np.nan)
        for i in range(self.n_states):
            if not self._calc_grad[i]:
                continue
            grad[i] = np.genfromtxt(f"SCIgradient_state{i+1}.txt")
        return grad

    def read_nac(self):
        nac = np.full((self._nstates, self._nstates, self._natoms, 3), np.nan)
        for i in range(self.n_states):
            for j in range(i):
                if not self._calc_nac[i,j]:
                    continue
                nac[i,j] = np.genfromtxt(f"NAC_state{j+1}state{i+1}.txt")
                nac[j,i] = nac[i,j]
        return nac

    def read_dipmom(self):
        raise NotImplementedError
        dipmom = np.full((self._nstates, self._nstates, 3), np.nan)
        return dipmom

    # probably change args
    def read_ovlp(self, atoms, geom1, geom2):
        raise NotImplementedError