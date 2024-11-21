import numpy as np

from classes.out import Output
from .sh import SurfaceHopping
from .checker import HoppingUpdater
from classes.molecule import Molecule, MoleculeBloch
from electronic.electronic import ESTProgram
from updaters.coeff import BlochUpdater
from updaters.tdc import TDCUpdater

class MISH(SurfaceHopping, key = "mish"):
    ''' Runeson and Manolopoulos "Multi Mash". Also known as "MISH", the artist previously known as SHIAM '''
    def __init__(self, **config):
        super().__init__(**config)

        if self._rescale != "mish":
            #            out = Output()
            #out.write_log(f"MISH called without MISH rescaling. Changing to default MISH rescaling\t")
            self._rescale = "mish"

        HoppingUpdater(key = "mish", **config["quantum"])

    def update_quantum(self, mols: Molecule):
        super().update_quantum(mols)
        self.update_target(mols)

    def adjust_nuclear(self, mols: list[Molecule]):
        out = Output()
        mol = mols[-1]
        #self._decoherence(mol, self._dt)

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
                if self._reverse:
                    out.write_log(f"Reversing along vector = {self._rescale}")
                    self._reverse_velocity(mol, delta)
                self._nohop()



    def population(self, mol: Molecule, s: int):
        N = mol.n_states
        H_N = np.sum(1/(np.arange(N)+1))
        a_N = (N-1)/(H_N-1) 
        return 1/N + a_N*(np.abs(mol.coeff_s[s])**2-1/N)



    def prepare_traj(self, mol: Molecule):
        ''' UPDATE '''
        nst = mol.n_states
        super().prepare_traj(mol)

        def _uniform_cap_distribution(nst: int, init_state: int):
            while True:
                a = np.random.normal(size=nst*2)
                ij = np.array([1.,1.j])
                coeff = np.sum(a.reshape((nst,2)) * ij[None,:],axis=1)
                coeff /= np.sqrt(np.sum(np.abs(coeff)**2))
                if (np.abs(coeff)**2).argmax() == init_state:
                    break
            return coeff

        coeff = _uniform_cap_distribution(nst,self.active)
        out = Output()
        out.write_log(f"Uniform cap initial conditions\t\tInitial state:      {self.active},\t\tInitial coeff:     {coeff}")
        out.write_log("\n")

        mol.coeff_s = coeff

    def _get_delta(self, mol: Molecule):
        ''' UPDATE '''

        def normalise(a):
            return a / np.linalg.norm(a)

        if "n" not in self.mode:
            self.setup_est(mode = "n")
            est = ESTProgram()
            est.run(mol)
            est.read(mol, mol)
            est.reset_calc()

        nst = mol.n_states
        coeff = mol.coeff_s
        d = mol.nacdr_ssad
        a = self.active
        target = self.target

        delta = np.zeros_like(mol.vel_ad)


        for i in range(nst):
            delta += np.real(np.conj(coeff[i])*d[i,a]*coeff[a] - np.conj(coeff[i])*d[i,target]*coeff[target])

        delta /= mol.mass_a[:,None]

        delta = normalise(delta)

        return delta















