import numpy as np
from .checker import HoppingUpdater
from dynamics.base import Dynamics
from classes.molecule import Molecule, SHMixin
from classes.out import Printer, Output
from classes.trajectory import Trajectory
from electronic.base import ESTMode
from updaters.composite import CompositeIntegrator
from updaters.coeff import CoeffUpdater
from updaters.tdc import TDCUpdater


class SurfaceHopping(Dynamics):
    mode = ESTMode("a")

    def __init__(self, *, dynamics: dict, **config):
        super().__init__(dynamics=dynamics, **config)
        dectypes = {
            "none": self._decoherence_none,
            "edc": self._decoherence_edc,
        }

        self._rescale = dynamics.get("rescale", "")
        self._reverse = dynamics.get("reverse", False)
        self._decoherence = dectypes[dynamics.get("decoherence", "none")]

    def step_mode(self, mol):
        return self.mode(mol) + TDCUpdater().mode(mol) + CoeffUpdater().mode(mol)

    def calculate_acceleration(self, mol: Molecule):
        mol.acc_ad = -mol.grad_sad[mol.active] / mol.mass_a[:,None]

    def potential_energy(self, mol: Molecule):
        return mol.ham_eig_ss[mol.active, mol.active]

    # move the switch elsewhere
    def _get_delta(self, mol: Molecule):
        # normalises vector
        def normalise(a):
            return a / np.linalg.norm(a)

        if self._rescale == "nac":
            print("Rescaling")
            # rescale along nacdr
            if np.any(np.isnan(mol.nacdr_ssad[mol.active, mol.target])):
                self.run_est(mol, ref = mol, mode = ESTMode("t")(mol))

            # check this works
            delta = mol.nacdr_ssad[mol.active, mol.target]
            delta /= mol.mass_a[:,None]
            delta = normalise(delta)
        elif self._rescale == "eff":
            delta = normalise(self._eff_nac(mol))
        else:
            # rescale uniformly
            delta = normalise(mol.vel_ad)
        return delta

    def _avail_kinetic_energy(self, mol: Molecule, delta: np.ndarray):
        a, b, _ = self._get_ABC(mol, delta)
        return b**2/(4*a)

    def _has_energy(self, mol: Molecule, delta: np.ndarray):
        return self._avail_kinetic_energy(mol, delta) + mol.ham_eig_ss[mol.active, mol.active] - mol.ham_eig_ss[mol.target, mol.target] > 0

    def _get_ABC(self, mol : Molecule, delta: np.ndarray):
        #to use later
        ediff =  mol.ham_eig_ss[mol.target, mol.target] - mol.ham_eig_ss[mol.active, mol.active]
        a = 0.5 * np.einsum('a,ai->',mol.mass_a, delta**2)
        b = -np.einsum('a,ai,ai->',mol.mass_a, mol.vel_ad, delta)
        c = ediff
        return a, b, c

    def _adjust_velocity(self, mol: Molecule, delta: np.ndarray):
        ediff =  mol.ham_eig_ss[mol.target, mol.target] - mol.ham_eig_ss[mol.active, mol.active]

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
        ediff =  mol.ham_eig_ss[mol.target, mol.target] - mol.ham_eig_ss[mol.active, mol.active]

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
            print(self._avail_kinetic_energy(mol,delta))
            print(mol.vel_ad * delta, np.sum((mol.vel_ad*delta)**2)/2,np.sum(mol.vel_ad/np.linalg.norm(mol.vel_ad)*delta/np.linalg.norm(delta)))
            print(mol.vel_ad,'\n',delta,'\n',a,b,c,D)
            raise RuntimeError("Issue with rescaling...")

        mol.vel_ad -= gamma * delta

        CompositeIntegrator().to_init()

    def _decoherence_none(*args):
        pass

    def _decoherence_edc(self, mol: Molecule, dt, c = 0.1):
        kin_en = mol.kinetic_energy
        for s in range(mol.n_states):
            if s == mol.active:
                continue
            else:
                decay_rate = 1/np.abs(mol.ham_eig_ss[s,s] - mol.ham_eig_ss[mol.active, mol.active]) * (1 + c/kin_en)
                mol.coeff_s[s] *= np.exp(-dt/decay_rate)

        tot_pop = np.sum(np.abs(mol.coeff_s)**2) - np.abs(mol.coeff_s[mol.active])**2
        mol.coeff_s[mol.active] *= np.sqrt(1 - tot_pop)/np.abs(mol.coeff_s[mol.active])

    def steps_elapsed(self, steps):
        super().steps_elapsed(steps)
        HoppingUpdater().elapsed(steps)

    def update_target(self, mols: list[Molecule], dt: float):
        hop = HoppingUpdater()
        hop.run(mols, dt)
        mols[-1].target = hop.hop.out

    # TODO: move to molecule
    def dat_header(self, traj: Trajectory):
        dic = super().dat_header(traj)
        dic["act"] = Printer.write("Active State", "s")
        return dic

    def dat_dict(self, traj: Trajectory):
        dic = super().dat_dict(traj)
        dic["act"] = Printer.write(traj.mol.active, "i")
        return dic

    def h5_dict(self, traj: Trajectory):
        dic = super().h5_dict(traj)
        dic["act"] = traj.mol.active
        return dic
