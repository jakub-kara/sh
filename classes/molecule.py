import numpy as np
from copy import deepcopy
from classes.config import Config, get_cfg
from classes.constants import convert, atomic_masses
from classes.format import fmt, cfmt

@Config("n_states", default=1)
@Config("n_atoms", default=None, valid=Config.NONE | Config.POSITIVE(Config.INT))
@Config("n_dim", default=None, valid=Config.NONE | Config.POSITIVE(Config.INT))
@Config("geom", path=Config.NUCLEAR, valid=Config.STR)
@Config("com", default=False, path=Config.NUCLEAR, valid=Config.BOOL)
@Config("vxyz", default=True, path=Config.NUCLEAR, valid=Config.BOOL)
@Config("active", default=None, path=Config.NUCLEAR, valid=Config.NONE | Config.LIST[Config.POSITIVE(Config.INT)])
@Config("inactive", default=None, path=Config.NUCLEAR, valid=Config.NONE | Config.LIST[Config.POSITIVE(Config.INT)])
class Molecule:
    def __init__(self, **config):
        inp = get_cfg(self, "geom", config)
        if inp is None:
            n_atoms, n_dim = get_cfg(self, ("n_atoms", "n_dim"), config)
            if n_atoms is None or n_dim is None:
                raise RuntimeError("Molecule requires *geom*, or *n_dim* and *n_atoms* to be specified.")

            self.pos_ad = np.zeros((n_atoms, n_dim))
            self.vel_ad = np.zeros((n_atoms, n_dim))
            self.acc_ad = np.zeros((n_atoms, n_dim))
            self.name_a = np.zeros(n_atoms, dtype="S2")
            self.mass_a = np.zeros(n_atoms)
        else:
            vxyz = get_cfg(self, "vxyz", config)
            if vxyz:
                self.from_vxyz(inp)
            else:
                self.from_xyz(inp)

        com = get_cfg(self, "com", config)
        if com:
            self.set_com()

        n_states = get_cfg(self, "n_states", config)
        self.ham_eig_ss = np.zeros((n_states, n_states))
        self.ham_dia_ss = np.zeros((n_states, n_states), dtype=np.complex128)
        self.trans_ss = np.eye(n_states, dtype=np.complex128)
        self.grad_sad = np.zeros((n_states, self.n_atoms, self.n_dim))
        self.nacdr_ssad = np.zeros((n_states, n_states, self.n_atoms, self.n_dim))
        self.nacdt_ss = np.zeros((n_states, n_states))
        self.ovlp_ss = np.eye(n_states)
        self.dipmom_ssd = np.zeros((n_states, n_states, self.n_dim))
        self.nacflp_ss = np.zeros((n_states, n_states))
        self.phase_s = np.ones(n_states)
        self.coeff_s = np.zeros(n_states, dtype=np.complex128)

        act = get_cfg(self, "active", config)
        inact = get_cfg(self, "inactive", config)

        if act is not None and inact is not None:
            raise LookupError("Both active and inactive modes provided.\nRemove one to proceed.")

        rev = False
        if inact is not None:
            act = inact
            rev = True
        if act is None:
            act = []

        if len(act) != self.n_dof:
            # assume a list of indices
            # check for duplicates
            if len(act) != len(set(act)):
                raise LookupError(f"Invalid format for {rev*'in'}active modes.\nUse a list of booleans a list of indices of {rev*'in'}active modes.")
            temp = np.zeros(self.n_dof)
            temp[act] = 1
            act = temp
        act = np.array(act)

        if rev:
            act = 1 - act

        self.act_dof = act.astype(bool).reshape((self.n_atoms, self.n_dim))
        if not np.any(self.act_dof):
            self.act_dof = np.logical_not(self.act_dof)

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
    def inact_dof(self):
        return np.logical_not(self.act_dof)

    @property
    def n_states(self):
        return self.ham_eig_ss.shape[0]

    @property
    def denmat_ss(self):
        return np.outer(self.coeff_s, np.conj(self.coeff_s))

    @property
    def mom_ad(self):
        return self.mass_a[:, None] * self.vel_ad

    @property
    def force_ad(self):
        return self.mass_a[:, None] * self.acc_ad

    @property
    def nac_norm_ss(self):
        return np.sqrt(np.sum(self.nacdr_ssad**2, axis=(2,3)))

    def kinetic_energy(self):
        return 0.5 * np.sum(self.mass_a[:,None] * self.vel_ad**2)

    def potential_energy(self):
        pass

    def total_energy(self):
        pass

    def population(self):
        pass

    def copy_empty(self):
        pass

    # might copy less
    def copy(self):
        return deepcopy(self)

    def eff_nac(self):
        nac_eff = np.zeros_like(self.nacdr_ssad)
        for i in range(self.n_states):
            for j in range(i):
                diff = self.grad_sad[i] - self.grad_sad[j]
                if np.abs(self.nacdt_ss[i,j]) < 1e-8:
                    alpha = 0
                else:
                    alpha = (self.nacdt_ss[i,j] - np.sum(diff * self.vel_ad)) / np.sum(self.vel_ad**2)
                nac_eff[i,j] = diff + alpha * self.vel_ad
                nac_eff[j,i] = -nac_eff[i,j]
        return nac_eff

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
        com_pos = np.sum(self.pos_ad * self.mass_a[:, None], axis=0) / total_mass
        self.pos_ad -= com_pos

        # com velocity
        com_vel = np.sum(self.vel_ad * self.mass_a[:, None], axis=0) / total_mass
        self.vel_ad -= com_vel

        # com rotation
        if self.n_dim != 3 or self.n_atoms < 3:
            return
        tot = np.zeros(3)
        for a in range(self.n_atoms):
            mom = self.vel_ad[a] * self.mass_a[a]
            ang = np.cross(mom, self.pos_ad[a])
            tot -= ang

        ang_vel = np.linalg.inv(self.inertia) @ tot
        for a in range(self.n_atoms):
            v_rot = np.cross(ang_vel, self.pos_ad[a])
            self.vel_ad[a] -= v_rot

    def adjust_ovlp(self):
        for i in range(self.n_states):
            self.ovlp_ss[i,:] *= self.phase_s[i]

        phase_vec = np.ones(self.n_states)
        for i in range(self.n_states):
            if self.ovlp_ss[i,i] < 0:
                phase_vec[i] *= -1
                self.ovlp_ss[:,i] *= -1
        self.phase_s = phase_vec

        # self.ovlp_ss /= np.sum(self.ovlp_ss**2, axis=0)

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

    def dat_header(self):
        nst = self.n_states
        dic = {
            "pop": "".join([fmt(f"Population {i}") for i in range(nst)]),
            "pes": "".join([fmt(f"Pot En {i} [eV]") for i in range(nst)]),
            "ken": fmt("Total Kin En [eV]"),
            "pen": fmt("Total Pot En [eV]"),
            "ten": fmt("Total En [eV]"),
            "nacdr": "".join([fmt(f"NACdr {j}-{i} [au]") for i in range(nst) for j in range(i)]),
            "nacdt": "".join([fmt(f"NACdt {j}-{i} [au]") for i in range(nst) for j in range(i)]),
            "coeff": "".join([s for i in range(nst) for s in [fmt(f"Re Coeff {i}"), fmt(f"Im Coeff {i}")]]),
        }
        dic |= {f"x-{i}": fmt(f"Position {i} [au]") for i in range(self.n_dof) if self.act_dof.flatten()[i]}
        dic |= {f"p-{i}": fmt(f"Momentum {i} [au]") for i in range(self.n_dof) if self.act_dof.flatten()[i]}
        return dic

    def dat_dict(self):
        nst = self.n_states
        dic = {
            "pop": "".join([fmt(self.population(i)) for i in range(nst)]),
            "pes": "".join([cfmt(self.ham_eig_ss[i,i], "au", "ev") for i in range(nst)]),
            "ken": cfmt(self.kinetic_energy(), "au", "ev"),
            "pen": cfmt(self.potential_energy(), "au", "ev"),
            "ten": cfmt(self.total_energy(), "au", "ev"),
            "nacdr": "".join([fmt(self.nac_norm_ss[i,j]) for i in range(nst) for j in range(i)]),
            "nacdt": "".join([fmt(self.nacdt_ss[i,j]) for i in range(nst) for j in range(i)]),
            "coeff": "".join([fmt(self.coeff_s[i]) for i in range(nst)]),
        }
        dic |= {f"x-{i}": fmt(x) for i, x in enumerate(self.pos_ad.flatten()) if self.act_dof.flatten()[i]}
        dic |= {f"p-{i}": fmt(x) for i, x in enumerate(self.mom_ad.flatten()) if self.act_dof.flatten()[i]}
        return dic

    def h5_dict(self):
        dic = {
            "pos": self.pos_ad,
            "vel": self.vel_ad,
            "acc": self.acc_ad,
            "trans": self.trans_ss,
            "hdiag": self.ham_eig_ss,
            "grad": self.grad_sad,
            "nacdr": self.nacdr_ssad,
            "nacdt": self.nacdt_ss,
            "coeff": self.coeff_s,
            "kinen": self.kinetic_energy(),
            "poten": self.potential_energy(),
            "toten": self.total_energy(),
        }
        return dic


@Config("mixin", valid = Config.LIST[Config.STR])
@Config("initstate", path=Config.DYNAMICS, valid=Config.POSITIVE(Config.INT))
@Config("qinit", default=None, path=Config.QUANTUM, valid=Config.STR | Config.NONE)
class MoleculeMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state = get_cfg(self, "initstate", kwargs)
        self._inp = get_cfg(self, "qinit", kwargs)

    def set_coeff(self):
        if self._inp is None:
            self.default_coeff()
        else:
            self.read_coeff(self._inp)

    def default_coeff(self):
        self.coeff_s[self._state] = 1.

    def read_coeff(self, file):
        data = np.genfromtxt(file)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape != (self.n_states, 2):
            raise ValueError(f"Invalid coeff input format in {file}")
        self.coeff_s[:] = data[:,0]
        self.coeff_s += 1j*data[:,1]

    def dat_header(self):
        return super().dat_header()

    def dat_dict(self):
        return super().dat_dict()

    def h5_dict(self):
        return super().h5_dict()

@Config.Option("sh", "mixin")
class SHMixin(MoleculeMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active = self._state
        self.target = self._state

    def hop_ready(self):
        return self.active != self.target

    def hop(self):
        self.active = self.target

    def nohop(self):
        self.target = self.active

    def dat_header(self):
        dic = {"act": fmt("Active")}
        return dic | super().dat_header()

    def dat_dict(self):
        dic = {"act": fmt(self.active)}
        return dic | super().dat_dict()

    def h5_dict(self):
        dic = {"act": self.active}
        return dic | super().h5_dict()

@Config.Option("bloch", "mixin")
class BlochMixin(SHMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bloch_n3 = np.zeros((self.n_states, 3))

    @property
    def coeff_s(self):
        # doi: 10.1063/5.0208575
        nst = self.n_states
        out = np.zeros(nst, dtype=np.complex128)

        # active state
        temp = 1
        for i in range(nst):
            if i == self.active:
                continue
            temp += (1 - self.bloch_n3[i,2]) / (1 + self.bloch_n3[i,2])
        out[self.active] = 1 / np.sqrt(temp)

        # other states magnitude
        for i in range(nst):
            if i == self.active:
                continue
            out[i] = out[self.active] * np.sqrt((1 - self.bloch_n3[i,2]) / (1 + self.bloch_n3[i,2]))

        # other states phase
        for i in range(nst):
            if i == self.active:
                continue
            arg = 0
            if np.abs(out[i]) > 1e-12:
                arg = np.angle(1/2 * (self.bloch_n3[i,0] + 1j*self.bloch_n3[i,1]) * (np.abs(out[self.active])**2 + out[i]**2) / (out[self.active] * out[i]))
            out[i] *= np.exp(1j*arg)

        return out

    @coeff_s.setter
    def coeff_s(self, val):
        # to be consistent with the rest of the code
        pass

    def default_coeff(self):
        self.bloch_n3[:, 2] = 1
        self.bloch_n3[self.active, :] = None

    def read_coeff(self, file):
        data = np.genfromtxt(file)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape != (self.n_states - 1, 3):
            raise ValueError(f"Invalid bloch input format in {file}")
        self.bloch_n3[:self.active] = data[:self.active]
        self.bloch_n3[self.active + 1:] = data[self.active:]

    def h5_dict(self):
        dic = {"bloch": self.bloch_n3}
        return dic | super().h5_dict()

@Config.Option("mmst", "mixin")
@Config("qinit", fix=None)
class MMSTMixin(MoleculeMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nst = self.n_states
        self.x_s = np.zeros(nst)
        self.p_s = np.zeros(nst)
        self.dxdt_s = np.zeros(nst)
        self.dpdt_s = np.zeros(nst)
        self.dRdt_s = np.zeros((self.n_atoms,3))
        self.dPkindt_s = np.zeros((self.n_atoms,3))

    @property
    def r2(self):
        return self.x_s**2 + self.p_s**2

    def read_coeff(self, file):
        data = np.genfromtxt(file)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape != (self.n_states, 2):
            raise ValueError(f"Invalid coeff input format in {file}")
        self.x_s[:] = data[:,0]
        self.p_s[:] = data[:,1]

@Config.Option("csdm", "mixin")
class CSDMMixin(MoleculeMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active = self._state
        self.coeff_co_s = np.zeros(self.n_states, dtype=np.complex128)
        self.coeff_co_s[:] = self.coeff_s

    @property
    def pointer(self):
        return self.active

    @pointer.setter
    def pointer(self, val):
        self.active = val