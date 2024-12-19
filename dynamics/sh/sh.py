import numpy as np
from .checker import HoppingUpdater
from dynamics.dynamics import Dynamics
from classes.molecule import Molecule
from classes.out import Printer, Output
from electronic.electronic import ESTProgram

class SurfaceHopping(Dynamics):
    mode = "a"

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)
        self._active = dynamics["initstate"]
        self._target = self._active
        dectypes = {
            "none": self._decoherence_none,
            "edc": self._decoherence_edc,
        }

        self._rescale = dynamics.get("rescale", "")
        self._reverse = dynamics.get("reverse", False)
        self._decoherence = dectypes[dynamics.get("decoherence", "none")]

    @property
    def active(self):
        return self._active

    @property
    def target(self):
        return self._target

    def hop_ready(self):
        return self.active != self.target

    @property
    def rescaling(self):
        return self._rescale

    def calculate_acceleration(self, mol: Molecule):
        mol.acc_ad = -mol.grad_sad[self.active] / mol.mass_a[:,None]

    def potential_energy(self, mol: Molecule):
        return mol.ham_eig_ss[self.active, self.active]

    @property
    def mode(self):
        return "a" + super().mode

    def setup_est(self, mode: str):
        est = ESTProgram()
        if "a" in mode:
            est.add_grads(self.active)
        if "g" in mode:
            est.all_grads()
        if "o" in mode:
            est.add_ovlp()
        if "n" in mode:
            est.all_nacs()

    # move the switch elsewhere
    def _get_delta(self, mol: Molecule):
        # normalises vector
        def normalise(a):
            return a / np.linalg.norm(a)

        if self._rescale == "nac":
            # rescale along nacdr
            if "n" not in self.mode:
                self.setup_est(mode = "n")
                est = ESTProgram()
                est.run(mol)
                est.read(mol, mol)
                est.reset_calc()

            # check this works
            delta = mol.nacdr_ssad[self.active, self.target]
            delta /= mol.mass_a[:,None]
            delta = normalise(delta)
        elif self._rescale == "eff":
            delta = normalise(self._eff_nac(mol))
        else:
            # rescale uniformly
            delta = normalise(mol.vel_ad)
        return delta

    def _avail_kinetic_energy(self, mol: Molecule, delta: np.ndarray):
        vel_proj = np.sum(mol.vel_ad * delta) * delta
        kin_proj = 0.5*np.sum(mol.mass_a[:,None]*vel_proj**2)
        return kin_proj

    def _has_energy(self, mol: Molecule, delta: np.ndarray):
        return self._avail_kinetic_energy(mol,delta) + mol.ham_eig_ss[self.active, self.active] - mol.ham_eig_ss[self.target, self.target] > 0

    def _hop(self):
        self._active = self.target
        self._recalc = True
        # SET INTEGRATOR TO INIT STATE

    def _nohop(self):
        self._target = self.active

    def _get_ABC(self, mol : Molecule, delta: np.ndarray):
        #to use later
        ediff =  mol.ham_eig_ss[self.target, self.target] - mol.ham_eig_ss[self.active, self.active] 
        a = 0.5 * np.einsum('a,ai->',mol.mass_a, delta**2)
        b = -np.einsum('a,ai,ai->',mol.mass_a, mol.vel_ad, delta)
        c = ediff
        return a, b, c

    def _adjust_velocity(self, mol: Molecule, delta: np.ndarray):
        ediff =  mol.ham_eig_ss[self.target, self.target] - mol.ham_eig_ss[self.active, self.active] 

        # compute coefficients in the quadratic equation
        a = 0.5 * np.einsum('a,ai->',mol.mass_a, delta**2)
        b = -np.einsum('a,ai,ai->',mol.mass_a, mol.vel_ad, delta)
        c = ediff

        # find the determinant
        D = b**2 - 4 * a * c
        if D < 0:
            print('Issue with rescaling...')
            raise RuntimeError
            # if self._reverse:
            #     gamma = -b/a
            # else:
            #     gamma = 0
        # choose the smaller solution of the two
        elif b < 0:
            gamma = -(b + np.sqrt(D)) / (2 * a)
        elif b >= 0:
            gamma = -(b - np.sqrt(D)) / (2 * a)

        mol.vel_ad -= gamma * delta

    def _reverse_velocity(self, mol: Molecule, delta: np.ndarray):
        ediff =  mol.ham_eig_ss[self.target, self.target] - mol.ham_eig_ss[self.active, self.active] 

        # compute coefficients in the quadratic equation
        a = 0.5 * np.einsum('a,ai->',mol.mass_a, delta**2)
        b = -np.einsum('a,ai,ai->',mol.mass_a, mol.vel_ad, delta)
        c = ediff

        # find the determinant
        D = b**2 - 4 * a * c
        if D < 0:
            # reverse if no real solution and flag set
            gamma = -b/a
        else:
            print('Issue with rescaling...')
            # print(self._avail_kinetic_energy(mol,delta))
            # print(mol.vel_ad * delta, np.sum((mol.vel_ad*delta)**2)/2,np.sum(mol.vel_ad/np.linalg.norm(mol.vel_ad)*delta/np.linalg.norm(delta)))
            # print(mol.vel_ad,'\n',delta,'\n',a,b,c,D)
            raise RuntimeError

        mol.vel_ad -= gamma * delta

    def _decoherence_none(*args):
        pass

    def _decoherence_edc(self, mol: Molecule, dt, c = 0.1):
        kin_en = mol.kinetic_energy
        for s in range(mol.n_states):
            if s == self.active:
                continue
            else:
                decay_rate = 1/np.abs(mol.ham_eig_ss[s,s] - mol.ham_eig_ss[self.active, self.active]) * (1 + c/kin_en)
                mol.coeff_s[s] *= np.exp(-dt/decay_rate)

        tot_pop = np.sum(np.abs(mol.coeff_s)**2) - np.abs(mol.coeff_s[self.active])**2
        mol.coeff_s[self.active] *= np.sqrt(1 - tot_pop)/np.abs(mol.coeff_s[self.active])

    def prepare_traj(self, mol):
        out = Output()
        out.write_log(f"Initial state:      {self.active}")
        mol.coeff_s[self.active] = 1
        out.write_log("\n")
        super().prepare_traj(mol)

    def update_target(self, mols: list[Molecule], dt: float):
        hop = HoppingUpdater()
        hop.elapsed(self._step)
        hop.run(mols, dt, self.active)
        self._target = hop.hop.out

    def dat_header(self, dic: dict, record: list):
        for rec in record:
            if rec == "act":
                dic[rec] += Printer.write("Active State", "s")
        return dic

    def dat_dict(self, dic: dict, record: list):
        for rec in record:
            if rec == "act":
                dic[rec] += Printer.write(self.active, "i")
        return dic
