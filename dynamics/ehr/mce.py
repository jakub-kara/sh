import numpy as np
from copy import deepcopy
from dynamics.ehr.ehr import SimpleEhrenfest
from classes.molecule import Molecule

class MultiEhrenfest(SimpleEhrenfest):
    def __init__(self, config: dict):
        super().__init__(config)
        
        dyn: dict = config["dynamics"]
        self._name = "multiconfigurational ehrenfest"
        self._dclone = dyn.get("dclone", 5e-6)
        self._dnac = dyn.get("dnac", 2e-3)
        self._maxspawn = dyn.get("maxspawn", 3)
        self._nspawn = 0
        inistate = dyn["initstate"]
        self._state = inistate
        self._accbr = np.zeros(self.n_states)
        self._phase = 0

    # TODO: symmetrise breaking force
    def _calculate_breaking(self, mol: Molecule):
        for s in range(mol.pes.n_states):
            dfbr = mol.pes.grad_sad[s] + mol.acc_ad * mol.mass_a[:,None]
            fbr = np.abs(mol.pes.coeff_s[s])**2 * dfbr
            self._accbr[s] = np.linalg.norm(fbr / mol.mass_a[:,None])

    def split_traj(self):
        temp = np.zeros_like(self.mol.pes.coeff_s)
        temp[self._split] = self.mol.pes.coeff_s[self._split]
        self.mol.pes.coeff_s[self._split] = 0
        self.mol.pes.coeff_s /= np.sqrt(1 - np.sum(np.abs(temp)**2))
        self._split = None
        
        clone = deepcopy(self)
        clone.mol.pes.coeff_s = temp
        clone.mol.pes.coeff_s /= np.sqrt(np.sum(np.abs(temp)**2))
        return clone

    def update_nuclear(self):
        self._phase += 0.5 * self.mol.kinetic_energy * self.dt
        super().update_nuclear()
        self._phase += 0.5 * self.mol.kinetic_energy * self.dt
 
    def adjust_nuclear(self):
        self._calculate_breaking(self.mol)
        # print(self._accbr)
        # print(np.abs(self.mol.pes.nacdt_ss[0,1]))
        
        # TODO: consider all subsets of states
        # https://doi.org/10.1021/acs.jctc.1c00131
        coeff = self.mol.pes.coeff_s
        mx = np.argmax(np.abs(coeff))
        # w = 1/np.sum(np.abs(coeff)**4)
        # cond1 = w > 1.3
        
        # fmean = -np.einsum("s,sad->ad", np.abs(coeff)**2, self.mol.pes.grad_sad)
        fmean = self.mol.acc_ad * self.mol.mass_a[:,None]
        fmax = -self.mol.pes.grad_sad[mx]
        theta = np.arccos((2 * np.sum(fmean * fmax)) / (np.sum(fmean**2) + np.sum(fmax**2)))
        cond2 = theta > np.pi/12

        delta = np.sum(np.abs(2 * np.real(coeff/coeff[mx]) * self.mol.pes.nacdt_ss[:,mx]))
        cond3 = delta < 5e-3

        print(theta, delta)        

        if cond2 and cond3 and self._nspawn < self._maxspawn:
            self._split = [mx]
            self._nspawn += 1

    # TODO: h5_info with widths

    def h5_dict(self):
        dic = super().h5_dict()
        dic["phase"] = self._phase
        return dic
    
    def write_outputs(self):
        super().write_outputs()