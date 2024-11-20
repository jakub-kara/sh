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

    def update_quantum(self, mols: Molecule):
        super().update_quantum(mols)
        self.update_target(mols)

    def adjust_nuclear(self, mols: list[Molecule]):
        out = Output()
        mol = mols[-1]
        self._decoherence(mol, self._dt)

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
                self._nohop()
                if self._reverse:
                    self._reverse_velocity(mol, delta)
