import numpy as np
from copy import deepcopy
from classes.pes import PES
from classes.constants import Constants

class Molecule:
    def __init__(self, *, n_states: int, input="geom.xyz", **config):
        self.pos_ad = None
        self.vel_ad = None
        self.acc_ad = None
        self.name_a = None
        self.mass_a = None
        self.from_vxyz(input)

        pes = PES(n_states, self.n_atoms)
        self._pes = pes

    @property
    def n_atoms(self):
        return self.mass_a.shape[0]

    def copy_empty(self):
        pass

    # might copy less
    def copy_all(self):
        return deepcopy(self)

    # probably should be static
    def from_xyz(self, filename: str):
        with open(filename, "r") as file:
            pos = []
            name = []
            mass = []
            for i, line in enumerate(file):
                if i == 0:
                    assert len(line.split()) == 1
                elif i == 1:
                    comment = line
                else:
                    line_list = line.split()
                    if len(line_list) > 0:
                        assert len(line_list) == 4, "wrong xyz file format"
                        name.append(line_list[0])
                        mass.append(Constants.atomic_masses[line_list[0].split('_')[0]]*Constants.amu)
                        pos.append([float(num.replace('d', 'e')) for num in line_list[1:4]])

            self.pos_ad = np.array(pos)
            self.vel_ad = np.zeros_like(self.pos_ad)
            self.acc_ad = np.zeros_like(self.pos_ad)
            self.name_a = np.array(name, dtype="S2")
            self.mass_a = np.array(mass)
        return self

    def to_xyz(self):
        outstr = f"{self.n_atoms}\n\n"
        pos = self.pos_ad * Constants.bohr2A
        name = self.name_a.astype("<U2")
        for i in range(self.n_atoms):
            outstr += f"{name[i]} {pos[i,0]} {pos[i,1]} {pos[i,2]}\n"
        return outstr

    def from_vxyz(self, filename: str):
        with open(filename, "r") as file:
            pos = []
            vel = []
            name = []
            mass = []
            for i, line in enumerate(file):
                if i == 0:
                    assert len(line.split()) == 1
                elif i == 1:
                    comment = line
                else:
                    line_list = line.split()
                    if len(line_list) > 0:
                        assert len(line_list) == 7, "wrong xyz file format"
                        name.append(line_list[0])
                        mass.append(Constants.atomic_masses[line_list[0].split('_')[0]]*Constants.amu)
                        pos.append([float(num.replace('d', 'e')) for num in line_list[1:4]])
                        vel.append([float(num.replace('d', 'e')) for num in line_list[4:7]])

            self.pos_ad = np.array(pos)
            self.vel_ad = np.array(vel)
            self.acc_ad = np.zeros_like(self.pos_ad)
            self.name_a = np.array(name, dtype="S2")
            self.mass_a = np.array(mass)
        return self

    def to_vxyz(self):
        outstr = f"{self.n_atoms}\n\n"
        pos = self.pos_ad * Constants.bohr2A
        vel = self.vel_ad * Constants.bohr2A / Constants.au2fs
        name = self.name_a.astype("<U2")
        for i in range(self.n_atoms):
            outstr += f"{name[i]} {pos[i,0]} {pos[i,1]} {pos[i,2]} {vel[i,0]} {vel[i,1]} {vel[i,2]}\n"
        return outstr

    def set_com(self):
        total_mass = np.sum(self.mass_a)

        # com position
        com_pos = np.sum(self.pos_ad * self.mass_a[:,None], axis=0) / total_mass
        self.pos_ad -= com_pos

        # com velocity
        com_vel = np.sum(self.vel_ad * self.mass_a[:,None], axis=0) / total_mass
        self.vel_ad -= com_vel

        # com rotation
        inertia = np.einsum("a,aij->ij", self.mass_a, np.einsum("ij,al->aij", np.eye(3), self.pos_ad**2) - np.einsum("ai,aj->aij", self.pos_ad, self.pos_ad))
        mom = np.sum(self.vel_ad * self.mass_a, axis=0)
        ang_mom = np.cross(self.pos_ad, mom)
        ang_vel = np.linalg.inv(inertia) @ ang_mom
        vel = np.cross(ang_vel, self.pos_ad)
        self.vel_ad -= vel

    @property
    def kinetic_energy(self):
        return 0.5 * np.sum(self.mass_a[:,None] * self.vel_ad**2)

    # Methods related to PES
    @property
    def n_states(self):
        return self._pes.n_states

    def bind_pes(self, pes: PES):
        self._pes = pes
        return self

    @property
    def pes(self):
        return self._pes