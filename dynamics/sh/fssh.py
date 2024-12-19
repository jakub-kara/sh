import numpy as np
from .sh import SurfaceHopping
from .checker import HoppingUpdater
from classes.molecule import Molecule
from classes.out import Output
from electronic.electronic import ESTProgram

class FSSH(SurfaceHopping, key = "fssh"):
    mode = "a"

    def __init__(self, *, dynamics, **config):
        super().__init__(dynamics=dynamics, **config)
        HoppingUpdater(key = dynamics["prob"], **config["quantum"])

    def adjust_nuclear(self, mols: list[Molecule], dt: float):
        out = Output()
        mol = mols[-1]
        self._decoherence(mol, dt)
        self.update_target(mols, dt)

        out.write_log(f"target: {self.target} \t\tactive: {self.active}")
        # print(f"Final pops: {np.abs(mol.coeff_s)**2}")
        # print(f"Check sum:  {np.sum(np.abs(mol.coeff_s)**2)}")
        if self.hop_ready():
            delta = self._get_delta(mol)
            if self._has_energy(mol, delta):
                out.write_log("Hop succesful")
                self._adjust_velocity(mol, delta)
                self._hop()
                out.write_log(f"New state: {self.active}")
                hop = HoppingUpdater()
                out.write_log(f"Integrated hopping probability: {np.sum(hop.prob.inter)}")

                self.setup_est(mode = "a")
                est = ESTProgram()
                est.run(mol)
                est.read(mol)
                self.calculate_acceleration(mol)
            else:
                out.write_log("Hop failed")
                out.write_log(f'vel: \n{mol.vel_ad}')
                out.write_log(f'delta: \n{delta}')
                out.write_log(f'available kinetic energy = {self._avail_kinetic_energy(mol,delta)}')
                out.write_log(f'energy difference {mol.ham_eig_ss[self.active, self.active] - mol.ham_eig_ss[self.target, self.target]}')
                if self._reverse:
                    self._reverse_velocity(mol, delta)
                self._nohop()
