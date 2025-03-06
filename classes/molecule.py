import numpy as np
from copy import deepcopy
from classes.meta import Factory, DynamicClassProxy
from classes.constants import convert, atomic_masses

class MoleculeFactory:
    _products: dict = {}

    @classmethod
    def create_molecule(cls, mixins, *args, **kwargs):
        if not isinstance(mixins, (list, tuple)):
            mixins = [mixins]
        name = "Molecule" + "".join(mixins)
        cls._products[name] = mixins
        mol = cls._get_molecule(name, mixins)
        setattr(cls, name, mol)
        return mol(*args, **kwargs)

    @classmethod
    def _get_molecule(cls, name, mixins):
        return type(
            name,
            tuple([MoleculeMixin[mix] for mix in mixins] + [Molecule]),
            {})

    @classmethod
    def save(cls):
        return cls._products.copy()

    @classmethod
    def restart(cls, dic: dict):
        cls._products = dic
        for key, val in cls._products.items():
            setattr(cls, key, cls._get_molecule(key, val))

class Molecule:
    def __init__(self, *, n_states: int, input="geom.xyz", **config):
        self.from_vxyz(input)

        self.ham_eig_ss = np.zeros((n_states, n_states))
        self.ham_dia_ss = np.zeros((n_states, n_states), dtype=np.complex128)
        self.trans_ss = np.eye(n_states, dtype=np.complex128)
        self.grad_sad = np.zeros((n_states, self.n_atoms, self.n_dim))
        self.nacdr_ssad = np.zeros((n_states, n_states, self.n_atoms, self.n_dim))
        self.nacdt_ss = np.zeros((n_states, n_states))
        self.ovlp_ss = np.zeros((n_states, n_states))
        self.dipmom_ssd = np.zeros((n_states, n_states, self.n_dim))
        self.nacflp_ss = np.zeros((n_states, n_states))
        self.phase_s = np.ones(n_states)
        self.coeff_s = np.zeros(n_states, dtype=np.complex128)

    @property
    def n_atoms(self):
        return self.pos_ad.shape[0]

    @property
    def n_dim(self):
        return self.pos_ad.shape[1]

    @property
    def n_dof(self):
        return self.n_dim * self.n_atoms

    @property
    def n_states(self):
        return self.ham_eig_ss.shape[0]

    @property
    def mom_ad(self):
        return self.mass_a[:, None] * self.vel_ad

    @property
    def force_ad(self):
        return self.mass_a[:, None] * self.acc_ad

    @property
    def nac_norm_ss(self):
        return np.sqrt(np.sum(self.nacdr_ssad**2, axis=(2,3)))

    def copy_empty(self):
        pass

    # might copy less
    def copy_all(self):
        return deepcopy(self)

    # probably should be static
    def from_xyz(self, filename: str):
        with open(filename, "r") as f:
            pos = []
            name = []
            mass = []

            data = f.readline().strip().split()
            assert len(data) == 1, "Wrong xyz format in line 1."
            nat = int(data[0])

            f.readline()

            for i in range(nat):
                data = f.readline().strip().split()
                if i == 0:
                    ndim = len(data) - 1
                else:
                    assert len(data) - 1 == ndim, f"Inconsistent number of dimensions in line {i+3}."
                temp = data[0].split('_')
                name.append(temp[0])
                if len(temp) > 1:
                    mass.append(convert(float(temp[1]), "amu", "au"))
                else:
                    mass.append(convert(atomic_masses[temp[0].upper()], "amu", "au"))
                pos.append([float(num.replace('d', 'e')) for num in data[1:]])

            self.pos_ad = np.array(pos)
            self.vel_ad = np.zeros_like(self.pos_ad)
            self.acc_ad = np.zeros_like(self.pos_ad)
            self.name_a = np.array(name, dtype="S2")
            self.mass_a = np.array(mass)
        return self

    def to_dist(self):
        outstr = ""
        for i in range(self.n_atoms):
            for j in range(self.n_dim):
                outstr += f"{self.pos_ad[i,j]:18.12f}"
        outstr += "\n"
        return outstr

    def to_xyz(self):
        outstr = f"{self.n_atoms}\n\n"
        pos = convert(self.pos_ad, "au", "aa")
        name = self.name_a.astype("<U2")
        for i in range(self.n_atoms):
            outstr += f"{name[i]}"
            for j in range(self.n_dim):
                outstr += f" {pos[i,j]}"
            outstr += "\n"
        return outstr

    def from_vxyz(self, filename: str):
        with open(filename, "r") as f:
            pos = []
            vel = []
            name = []
            mass = []

            data = f.readline().strip().split()
            assert len(data) == 1, "Wrong xyz format in line 1."
            nat = int(data[0])

            f.readline()

            for i in range(nat):
                data = f.readline().strip().split()
                if i == 0:
                    ndim = (len(data) - 1) // 2
                else:
                    assert (len(data) - 1) // 2 == ndim, f"Inconsistent number of dimensions in line {i+3}."
                temp = data[0].split('_')
                name.append(temp[0])
                if len(temp) > 1:
                    mass.append(convert(float(temp[1]), "amu", "au"))
                else:
                    mass.append(convert(atomic_masses[temp[0].upper()], "amu", "au"))
                pos.append([float(num.replace('d', 'e')) for num in data[1:ndim+1]])
                vel.append([float(num.replace('d', 'e')) for num in data[ndim+1:]])

            self.pos_ad = np.array(pos)
            self.vel_ad = np.array(vel)
            self.acc_ad = np.zeros_like(self.pos_ad)
            self.name_a = np.array(name, dtype="S2")
            self.mass_a = np.array(mass)
        return self

    def to_vxyz(self):
        outstr = f"{self.n_atoms}\n\n"
        pos = convert(self.pos_ad, "au", "aa")
        vel = convert(self.vel_ad, "au", "aa fs^-1")
        name = self.name_a.astype("<U2")
        for i in range(self.n_atoms):
            outstr += f"{name[i]}"
            for j in range(self.n_dim):
                outstr += f" {pos[i,j]}"
            for j in range(self.n_dim):
                outstr += f" {vel[i,j]}"
            outstr += "\n"
        return outstr

    @property
    def inertia(self):
        return np.einsum("a,aij->ij", self.mass_a, np.einsum("ij,al->aij", np.eye(3), self.pos_ad**2) - np.einsum("ai,aj->aij", self.pos_ad, self.pos_ad))

    def set_com(self):
        total_mass = np.sum(self.mass_a)

        # com position
        com_pos = np.sum(self.pos_ad * self.mass_a[:,None], axis=0) / total_mass
        self.pos_ad -= com_pos

        # com velocity
        com_vel = np.sum(self.vel_ad * self.mass_a[:,None], axis=0) / total_mass
        self.vel_ad -= com_vel

        # com rotation
        mom = np.sum(self.vel_ad * self.mass_a, axis=0)
        ang_mom = np.cross(self.pos_ad, mom)
        ang_vel = np.linalg.inv(self.inertia) @ ang_mom
        vel = np.cross(ang_vel, self.pos_ad)
        self.vel_ad -= vel

    @property
    def kinetic_energy(self):
        return 0.5 * np.sum(self.mass_a[:,None] * self.vel_ad**2)

    @property
    def dmat_ss(self):
        return np.outer(self.coeff_s.conj(), self.coeff_s)

    def adjust_ovlp(self):
        for i in range(self.n_states):
            self.ovlp_ss[i,:] *= self.phase_s[i]

        phase_vec = np.ones(self.n_states)
        for i in range(self.n_states):
            if self.ovlp_ss[i,i] < 0:
                phase_vec[i] *= -1
                self.ovlp_ss[:,i] *= -1
        self.phase_s = phase_vec

    def adjust_nacs(self, other):
        for s1 in range(self.n_states):
            for s2 in range(s1):
                # calculate overlap
                if np.sum(other.nacdr_ssad[s1,s2] * self.nacdr_ssad[s1,s2]) < 0:
                    # flip sign if overlap < 0
                    print("FLIP")
                    self.nacdr_ssad[s1,s2] = -self.nacdr_ssad[s1,s2]
                self.nacdr_ssad[s2,s1] = -self.nacdr_ssad[s1,s2]

    def adjust_tdc(self, other):
        for s1 in range(self.n_states):
            for s2 in range(s1):
                # calculate overlap
                if other.nacdt_ss[s1,s2] * self.nacdt_ss[s1,s2] < 0:
                    # flip sign if overlap < 0
                    self.nacdt_ss[s1,s2] = -self.nacdt_ss[s1,s2]
                self.nacdt_ss[s2,s1] = -self.nacdt_ss[s1,s2]

    def adjust_energy(self, refen: float):
        for s in range(self.n_states):
            self.ham_dia_ss[s,s] -= refen

    def diagonalise_ham(self):
        eval, evec = np.linalg.eigh(self.ham_dia_ss)
        self.trans_ss = evec
        self.ham_eig_ss = np.diag(eval)

    def transform(self, diagonalise: bool = False):
        if not diagonalise:
            self.ham_eig_ss[:] = np.real(self.ham_dia_ss)
            self.trans_ss[:] = np.eye(self.n_states)
            return

        # need to transform gradient for non-diagonal hamiltonian
        # for details, see https://doi.org/10.1002/qua.2489
        self.diagonalise_ham()
        g_diab = np.zeros_like(self.nacdr_ssad, dtype=np.complex128)
        for i in range(self.n_states):
            for j in range(self.n_states):
                # on-diagonal part
                g_diab[i,j] = (i == j) * self.grad_sad[i]
                # off-diagonal part
                g_diab[i,j] -= (self.ham_dia_ss[i,i] - self.ham_dia_ss[j,j]) * self.nacdr_ssad[i,j]
        # just a big matrix multiplication with some extra dimensions
        g_diag = np.einsum("ij,jkad,kl->ilad", self.trans_ss.conj().T, g_diab, self.trans_ss)

        # only keep the real part of the gradient
        for i in range(self.n_states):
            self.grad_sad[i] = np.real(g_diag[i,i])

    def __reduce__(self):
        return (DynamicClassProxy(), (MoleculeFactory, self.__class__.__name__), self.__dict__.copy())

    def to_dict(self):
        pass

class MoleculeMixin(metaclass = Factory):
    pass

class SHMixin(MoleculeMixin):
    key = "sh"

    def __init__(self, *, initstate: int, **kwargs):
        super().__init__(**kwargs)
        self.active = initstate
        self.target = initstate

    def hop_ready(self):
        return self.active != self.target

    def hop(self):
        self.active = self.target

    def nohop(self):
        self.target = self.active

class EhrMixin(MoleculeMixin):
    key = "ehr"

    def __init__(self, *, initstate: int, **kwargs):
        super().__init__(**kwargs)
        self.state = initstate

class MCEMixin(EhrMixin):
    key = "mce"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nspawn = 0
        self.phase = 0
        self.split = None

class BlochMixin(MoleculeMixin):
    key = "bloch"

    def __init__(self, *, n_states, **nuclear):
        super().__init__(n_states=n_states, **nuclear)
        self.bloch_n3 = np.zeros((n_states, 3))

class MMSTMixin(MoleculeMixin):
    key = "mmst"

    def __init__(self, *, initstate, n_states, **nuclear):
        super().__init__(n_states=n_states, **nuclear)
        self.state = initstate
        self.x_s = np.zeros(n_states)
        self.p_s = np.zeros(n_states)
        self.dxdt_s = np.zeros(n_states)
        self.dpdt_s = np.zeros(n_states)
        self.dRdt_s = np.zeros((self.n_atoms,3))
        self.dPkindt_s = np.zeros((self.n_atoms,3))

    @property
    def r2(self):
        return self.x_s**2 + self.p_s**2

class CSDMMixin(MoleculeMixin):
    key = "csdm"

    def __init__(self, *, initstate: int, n_states: int, **nuclear):
        super().__init__(n_states=n_states, **nuclear)
        self.pointer = initstate
        self.coeff_co_s = np.zeros(n_states, dtype=np.complex128)



