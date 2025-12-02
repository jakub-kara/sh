import numpy as np
from .sh import SurfaceHopping
from classes.config import Config, get_cfg
from classes.molecule import Molecule
from classes.out import FileManager as fm
from classes.trajectory import Trajectory
from dynamics.decoherence import Decoherence
from electronic.base import ESTMode, ESTRequest
from updaters.nuclear import CompositeIntegrator

@Config.Option("fssh", "method")
class FSSH(SurfaceHopping):
    _molmixins = ["sh"]

    def adjust_nuclear(self, traj: Trajectory):
        mol = traj.mol
        ts = traj.timestep
        dec = Decoherence()
        dec.run(traj.mols, ts)
        mol.coeff_s[:] = dec.out.out
        self.update_target(traj.mols, ts)

        fm.log(f"target: {mol.target} \t\tactive: {mol.active}")
        print(f"Final pops: {np.abs(mol.coeff_s)**2}")
        print(f"Check sum:  {np.sum(np.abs(mol.coeff_s)**2)}")
        if mol.hop_ready():
            delta = self._get_delta(mol)
            if self._has_energy(mol, delta):
                self.perform_hop(traj.mols, delta)
            else:
                fm.log("Hop failed")
                # out.write_log(f'vel: \n{mol.vel_ad}')
                # out.write_log(f'delta: \n{delta}')
                fm.log(f'available kinetic energy = {self._avail_kinetic_energy(mol,delta)}')
                fm.log(f'energy difference {mol.ham_eig_ss[mol.active, mol.active] - mol.ham_eig_ss[mol.target, mol.target]}')
                if self._reverse:
                    fm.log("Reversing velocity")
                    self._reverse_velocity(mol, delta)
                mol.nohop()
            fm.log()

    @ESTRequest(ESTMode(ESTMode.ACTIVE))
    def perform_hop(self, mols: list[Molecule], delta: float):
        fm.log("Hop succesful")

        mol = mols[-1]
        self._adjust_velocity(mol, delta)
        mol.hop()
        CompositeIntegrator().to_init()
        fm.log(f"New state: {mol.active}")

        self.run_est(mol, ref = mols[-2])
        self.calculate_acceleration(mol)