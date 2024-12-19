import numpy as np
from .ehr import SimpleEhrenfest
from classes.molecule import Molecule
from classes.out import Output, Printer

class BCMF(SimpleEhrenfest, key = "bcmf"):
    def _adjust_coeff(self, coeff: np.ndarray, selected: np.ndarray):
        tot = np.sum(np.abs(coeff)**2 * selected)
        coeff *= selected / np.sqrt(tot)

    def _choose_group(self, coeff: np.ndarray, refl: np.ndarray):
        tot = np.sum(np.abs(coeff)**2 * refl)
        out = Output()
        r = np.random.uniform()
        if r < tot:
            out.write_log("Chose reflected group")
        else:
            out.write_log("Chose non-reflected group")
        return r < tot

    def adjust_nuclear(self, mols: list[Molecule], dt: float):
        out = Output()
        mol = mols[-1]
        nst = mol.n_states
        eta = np.zeros(nst)
        allow = np.zeros(nst, dtype=bool)
        refl = np.zeros(nst, dtype=bool)

        for s in range(nst):
            temp = 1 + (self.potential_energy(mol) - mol.ham_eig_ss[s,s]) / mol.kinetic_energy
            allow[s] = temp > 0
            if temp > 0:
                eta[s] = np.sqrt(temp)
            else:
                eta[s] = 1e-10

        avg_mom = mol.mom_ad + dt * mol.force_ad
        mom_old = np.einsum("s, ad -> sad", eta, mol.mom_ad)
        mom_new = mom_old - dt * mol.grad_sad

        refl_avg = np.sum(mol.mom_ad * mol.force_ad) * np.sum(avg_mom * mol.force_ad) < 0
        out.write_log(f"MF      - reflect: {refl_avg}")
        for s in range(nst):
            refl[s] = np.sum(mom_old[s] * mol.grad_sad[s]) * np.sum(mom_new[s] * mol.grad_sad[s]) < 0
            out.write_log(f"State {s} - reflect: {refl[s]}      allowed: {allow[s]}")

        print(f"Init pops:  {np.abs(mol.coeff_s)**2}")
        print(f"Check sum:  {np.sum(np.abs(mol.coeff_s)**2)}")
        if refl_avg:
            prg = np.sum(np.abs(mol.coeff_s)**2 * allow)
            e_old = self.potential_energy(mol)
            mol.coeff_s *= allow / np.sqrt(prg)
            mol.vel_ad *= np.sqrt(1 + (e_old - self.potential_energy(mol)) / mol.kinetic_energy)

        elif np.any(refl):
            e_old = self.potential_energy(mol)
            out.write_log(Printer.write(e_old + mol.kinetic_energy, "f"))

            coeff = mol.coeff_s.copy()
            reflect = self._choose_group(mol.coeff_s, refl)
            self._adjust_coeff(mol.coeff_s, refl == reflect)
            print(refl == reflect)

            temp = 1 + (e_old - self.potential_energy(mol)) / mol.kinetic_energy
            if temp < 0:
                out.write_log("Chosen group not allowed")
                mol.coeff_s[:] = coeff
                self._adjust_coeff(mol.coeff_s, refl != reflect)

            mol.vel_ad *= np.sqrt(1 + (e_old - self.potential_energy(mol)) / mol.kinetic_energy)
            out.write_log(Printer.write(self.potential_energy(mol) + mol.kinetic_energy, "f"))

        self.calculate_acceleration(mol)

        print(f"Final pops: {np.abs(mol.coeff_s)**2}")
        print(f"Check sum:  {np.sum(np.abs(mol.coeff_s)**2)}")
