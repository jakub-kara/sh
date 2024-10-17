import numpy as np
from classes.molecule import Molecule
from classes.out import Printer
from electronic.electronic import ESTProgram
from integrators.tdc import TDCUpdater, LD
from dynamics.sh.sh import SurfaceHopping, HoppingChecker, HopUpdater

class FSSH(SurfaceHopping):
    # resolve
    def __init__(self, **config):
        super().__init__(**config)
        breakpoint()

    @property
    def active(self):
        return self._hopchck.active

    @property
    def target(self):
        return self._hopchck.target

    @property
    def rescaling(self):
        return self._rescale

    def calculate_acceleration(self, mol: Molecule):
        mol.acc_ad = -mol.pes.grad_sad[self.active] / mol.mass_a[:,None]

    def potential_energy(self, mol: Molecule):
        return mol.pes.ham_eig_ss[self.active, self.active]

    def _get_mode(self):
        return self._qupd.mode + "a"

    def setup_est(self, est: ESTProgram, mode: str):
        if "a" in mode:
            est.add_grads(self.active)
        if "g" in mode:
            est.all_grads()
        if "o" in mode:
            est.add_ovlp()
        if "n" in mode:
            est.all_nacs()

    def bind_components(self, *, electronic: dict, nuclear: dict, quantum: dict, output: dict, dynamics: dict, **config):
        self._seed = np.random.seed(dynamics.get("seed"))
        self.bind_est(**electronic)
        self.bind_nuclear_integrator(nuclear["nucupd"])
        self.bind_molecules(**nuclear)

        active = dynamics["initstate"]
        self._hopchck: HoppingChecker = select_hop_checker(dynamics["type"])(self.n_states, active)
        self.bind_quantum_updater(**quantum)
        self.bind_io(**output)

    def bind_quantum_updater(self, **quantum):
        qupd = HopUpdater(hop=self._hopchck, **quantum)
        self._qupd = qupd

    # move the switch elsewhere
    def _get_delta(self, mol: Molecule):
        # normalises vector
        def normalise(a):
            return a / np.linalg.norm(a)

        if self._rescale == 'ddr':
            # rescale along ddr
            delta = normalise(mol.pes.nacdr_ssad[self.active, self.target])
            delta /= mol.mass_a[:,None]
        elif self._rescale == 'mash':
            # rescale along expression E3 in https://doi.org/10.1063/5.0158147
            delta = np.zeros_like(mol.vel_ad)
            for i in range(mol.n_states):
                delta += np.real(np.conj(mol.pes.coeff_s[-1,i]) * mol.pes.nacdr_ssad[i, self.active] * mol.pes.coeff_s[-1, self.active] -
                                 np.conj(mol.pes.coeff_s[-1,i]) * mol.pes.nacdr_ssad[i, self.target] * mol.pes.coeff_s[-1, self.target])
            delta /= mol.mass_a[:,None]
            delta = normalise(delta)
        else:
            # rescale uniformly
            delta = normalise(mol.vel_ad)
        return delta

    def _avail_kinetic_energy(self, mol: Molecule):
        delta = self._get_delta(mol)
        a = np.einsum("ad, ad -> a", mol.vel_ad, delta)
        b = np.einsum("ad, ad -> a", delta, delta)
        return 0.5 * np.sum(mol.mass_a * a**2 / b)

    def _has_energy(self, mol: Molecule):
        return self._avail_kinetic_energy(mol) + self.potential_energy(mol) - mol.pes.ham_eig_ss[self.target, self.target] > 0

    def _hop(self):
        self._hopchck.hop_success()
        self._recalc = True
        # SET INTEGRATOR TO INIT STATE

    def _nohop(self):
        self._hopchck.hop_fail()

    def _adjust_velocity(self, mol: Molecule):
        delta = self._get_delta(mol)
        ediff = mol.pes.ham_eig_ss[self.active, self.active] - mol.pes.ham_eig_ss[self.target, self.target]

        # compute coefficients in the quadratic equation
        a = 0.5 * np.sum(mol.mass_a[:, None] * delta * delta)
        b = -np.sum(mol.mass_a[:, None] * mol.vel_ad * delta)
        c = -ediff

        # find the determinant
        D = b**2 - 4 * a * c
        if D < 0:
            # reverse if no real solution and flag set
            if self._reverse:
                gamma = -b/a
            else:
                gamma = 0
        # choose the smaller solution of the two
        elif b < 0:
            gamma = -(b + np.sqrt(D)) / (2 * a)
        elif b >= 0:
            gamma = -(b - np.sqrt(D)) / (2 * a)

        mol.vel_ad -= gamma * delta

    def _decoherence_none(*args):
        pass

    def _decoherence_edc(self, mol: Molecule, dt, c = 0.1):
        kin_en = mol.kinetic_energy
        for s in range(mol.n_states):
            if s == self.active:
                continue
            else:
                decay_rate = 1/np.abs(mol.pes.ham_eig_ss[s,s] - mol.pes.ham_eig_ss[self.active, self.active])*(1 + c/kin_en)
                mol.pes.coeff_s[s] *= np.exp(-dt/decay_rate)

        tot_pop = np.sum(np.abs(mol.pes.coeff_s)**2) - np.abs(mol.pes.coeff_s[self.active])**2
        mol.pes.coeff_s[self.active] *= np.sqrt(1 - tot_pop)/np.abs(mol.pes.coeff_s[self.active])

    def prepare_traj(self):
        self.mol.pes.coeff_s[self.active] = 1
        super().prepare_traj()

    def propagate(self):
        super().propagate()
        print(self._hopchck.prob_s)

    def adjust_nuclear(self):
        self._recalc = False
        self._decoherence(self.mol, self._dt)

        print(f"active: {self._hopchck.active}")
        print(f"target: {self._hopchck.target}")

        if self._hopchck.hop_ready():
            if self._has_energy(self.mol):
                self._adjust_velocity(self.mol)
                self._hop()
                self.run_est(self.mol, mode="a")
                self._nucupd.to_init()
            else:
                self._nohop()

    def dat_header(self, record):
        dic = super().dat_header(record)
        for rec in record:
            if rec == "act":
                dic[rec] += Printer.write("Active State", "s")
        return dic

    def dat_dict(self, record):
        dic = super().dat_dict(record)
        for rec in record:
            if rec == "act":
                dic[rec] += Printer.write(self.active, "i")
        return dic

def select_hop_checker(key: str):
    types = {
        "fs": FSSHHoppingChecker,
        "ld": LDHoppingChecker,
        "gf": GFHoppingChecker,
    }
    return types[key]

class FSSHHoppingChecker(HoppingChecker):
    def get_target(self, mol: list[Molecule], tdcupd: TDCUpdater, dtq: float, frac: float):
        act = self._active
        self._prob[:] = 0
        if self._target != act:
            return
        for s in range(self.n_states):
            # assign 0 hopping probability to active state
            if s == act:
                self._prob[s] = 0
            # standard Tully-based hopping probability
            else:
                # TODO: check timestep changes with variable timestep
                prob = np.real(tdcupd.interpolate(mol, frac)[s, act] * np.conj(mol[-1].pes.coeff_s[act]) * mol[-1].pes.coeff_s[s])
                prob *= -2 * dtq / np.abs(mol[-1].pes.coeff_s[act])**2
                self._prob[s] = max(0, prob)

        self._target = self._check_hop(act)

class LDHoppingChecker(HoppingChecker):
    def get_target(self, mol: list[Molecule], tdcupd: LD, *args, **kwargs):
        act = self._active
        for s in range(self.n_states):
            if s == act:
                self.prob_s[s] = 0
            else:
                prob = (1 - np.abs(mol[-1].pes.coeff_s[act])**2 / np.abs(mol[-2].pes.coeff_s[act])**2)
                prob *= np.real(mol[-1].pes.coeff_s[s] * np.conj(tdcupd.rmat[s, act]) * np.conj(mol[-2].pes.coeff_s[act]))
                prob /= (np.abs(mol[-2].pes.coeff_s[act])**2 - \
                         np.real(mol[-1].pes.coeff_s[act] * np.conj(tdcupd.rmat[act, act]) * np.conj(mol[-2].pes.coeff_s[act])))
                self.prob_s[s] = max(0, prob)
        self._target = self._check_hop(act)

class GFHoppingChecker(HoppingChecker):
    def get_target(self, mol: list[Molecule], *args, **kwargs):
        act = self._active
        div = np.sum(np.maximum(0, np.abs(mol[-2].pes.coeff_s)**2 - np.abs(mol[-1].pes.coeff_s)**2))
        for s in range(self.n_states):
            if s == act:
                self.prob_s[s] = 0
            else:
                prob = (1 - np.abs(mol[-1].pes.coeff_s[act])**2 / np.abs(mol[-2].pes.coeff_s[act])**2)
                prob *= np.abs(mol[-1].pes.coeff_s[s])**2 - np.abs(mol[-2].pes.coeff_s)**2
                prob /= div
                self.prob_s[s] = max(0, prob)
        self._target = self._check_hop(act)
