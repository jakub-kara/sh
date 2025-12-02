import numpy as np
import os
from abc import abstractmethod
from classes.config import Config, get_cfg
from classes.meta import Singleton, Counter, Decorator, DecoratorDistributor
from classes.molecule import Molecule
from classes.constants import convert

def est_method(func):
    func._estmethod = True
    return func

class ESTMode:
    NONE = ""
    ACTIVE = "a"
    GRADIENTS = "g"
    NACMES = "n"
    OVERLAP = "o"
    DIPOLE = "d"
    TARGET = "t"

    def __init__(self, val: str = ""):
        self.vals = [c.lower() for c in val]

    def __call__(self, mol: Molecule = None):
        out = []
        for mode in self.vals:
            if mode == "o":
                out.append("o")
            elif mode == "d":
                out.append("d")
            elif mode == "g":
                out.extend([f"g{i}" for i in range(mol.n_states)])
            elif mode == "a":
                out.append(f"g{mol.active}")
            elif mode == "n":
                out.extend([f"n{i}{j}" for i in range(mol.n_states) for j in range(i)])
            elif mode == "t":
                out.append(f"n{mol.active}{mol.target}")
        return out

class ESTRequest(Decorator):
    requests = []

    def __init__(self, *obj, this = True):
        self._obj = [i for i in obj if hasattr(i, "mode")]
        self._this = this
        self.id = None

    def run(self, func, instance, *args, **kwargs):
        req = []
        for obj in self._obj:
            if isinstance(obj, ESTMode):
                req.append(obj)
            elif hasattr(obj, "mode"):
                req.append(obj().mode)
        if self._this and hasattr(instance, "mode"):
            req.append(instance.mode)
        ESTRequest.requests.append(req)
        res = func(instance, *args, **kwargs)
        ESTRequest.requests.pop()
        return res

@Config("transform", default="none", path=Config.ELECTRONIC, valid=Config.STR)
class HamTransform(metaclass = Singleton):
    mode = ESTMode(ESTMode.NONE)

    @abstractmethod
    def transform(self, mol: Molecule):
        pass

    def diagonalise_ham(self, mol: Molecule):
        eval, evec = np.linalg.eigh(mol.ham_dia_ss)
        mol.trans_ss = evec
        mol.ham_eig_ss = np.diag(eval)

@Config.Option("none", "transform")
class NoTransform(HamTransform):
    def transform(self, mol: Molecule):
        mol.ham_eig_ss[:] = np.real(mol.ham_dia_ss)
        mol.trans_ss[:] = np.eye(mol.n_states)

@Config.Option("ngt", "transform")
class NGT(HamTransform):
    mode = ESTMode(ESTMode.GRADIENTS + ESTMode.NACMES)

    def transform(self, mol):
        # need to transform gradient for non-diagonal hamiltonian
        # for details, see https://doi.org/10.1002/qua.2489
        self.diagonalise_ham(mol)
        g_dia = np.zeros_like(mol.nacdr_ssad, dtype=np.complex128)
        for i in range(mol.n_states):
            for j in range(mol.n_states):
                # on-diagonal part
                g_dia[i,j] = (i == j) * mol.grad_sad[i]
                # off-diagonal part
                g_dia[i,j] -= (mol.ham_dia_ss[i,i] - mol.ham_dia_ss[j,j]) * mol.nacdr_ssad[i,j]
        # just a big matrix multiplication with some extra dimensions
        g_eig = np.einsum("ij,jkad,kl->ilad", mol.trans_ss.conj().T, g_dia, mol.trans_ss)

        # only keep the real part of the gradient
        for i in range(mol.n_states):
            mol.grad_sad[i] = np.real(g_eig[i,i])

@Config("program", path=Config.ELECTRONIC, valid=Config.STR)
@Config("path", default="", path=Config.ELECTRONIC, valid=Config.STR | Config.NONE)
@Config("method", default="", path=Config.ELECTRONIC, valid=Config.STR | Config.NONE)
@Config("states", path=Config.ELECTRONIC, valid=Config.LIST[Config.POSITIVE(Config.INT)])
@Config("refen", default=0, path=Config.ELECTRONIC, valid=Config.UNIT)
@Config("options", default=None, path=Config.ELECTRONIC, valid=Config.ANY)
class ESTProgram(DecoratorDistributor, metaclass = Singleton):
    _methods = {}

    def __init__(self, **config):
        self._register_methods()

        self._path = get_cfg(self, "path", config)
        states = get_cfg(self, "states", config)
        if isinstance(states, int):
            self._states = np.array([states])
        else:
            self._states = np.trim_zeros(np.array(states), "b")
        self._nstates = np.sum(self._states)
        self._natoms = None
        self._spinsum = np.cumsum(self._states) - self._states
        self._refen = convert(get_cfg(self, "refen", config), "au")

        self._method_name = get_cfg(self, "method", config)
        self._method = self._select_method(self._method_name)

        options = get_cfg(self, "options", config)
        if options is None:
            options = {}
        self._options = options
        self._file = self.__class__.__name__.lower()

        self._calc_grad = np.zeros(self._nstates)
        self._calc_nac = np.zeros((self._nstates, self._nstates))
        self._calc_ovlp = False
        self._calc_dip = False

    @property
    def n_states(self):
        return self._nstates

    def _register_methods(self):
        cls = self.__class__
        for key, val in cls.__dict__.items():
            if hasattr(val, "_estmethod"):
                cls._methods[key] = val

    def _select_method(self, key: str):
        return self._methods[key]

    def reset_calc(self):
        self._calc_grad = np.zeros(self._nstates)
        self._calc_nac = np.zeros((self._nstates, self._nstates))
        self._calc_ovlp = False
        return self

    def request(self, mol):
        if len(ESTRequest.requests) < 1:
            return

        reqs = ESTRequest.requests[-1]
        for req in reqs:
            args = req(mol)
            for arg in args:
                if arg[0] == "o":
                    self._calc_ovlp = True
                elif arg[0] == "g":
                    idx = int(arg[1])
                    self._calc_grad[idx] = True
                elif arg[0] == "n":
                    idx = int(arg[1]), int(arg[2])
                    self._calc_nac[idx] = True
                    self._calc_nac[idx[::-1]] = True
                elif arg[0] == "d":
                    self._calc_dip = True

    def any_grads(self):
        return np.any(self._calc_grad)

    def any_nacs(self):
        return np.any(self._calc_nac)

    def any_ovlp(self):
        return self._calc_ovlp

    def any_dip(self):
        return self._calc_dip

    @Counter(id = "est")
    def run(self, mol: Molecule):
        os.chdir("est")
        self.request(mol)
        self.write(mol)
        self.execute()
        os.chdir("..")

    def read(self, mol: Molecule, ref: Molecule = None):
        os.chdir("est")

        mol.ham_dia_ss = self.read_ham()
        mol.adjust_energy(self._refen)

        if self.any_nacs():
            if ref is None:
                raise RuntimeError("Cannot read nacmes without a reference Molecule.")
            mol.nacdr_ssad = self.read_nac()
            mol.adjust_nacs(ref)
        else:
            mol.nacdr_ssad[:] = None

        if self.any_grads():
            mol.grad_sad = self.read_grad()
        else:
            mol.grad_sad[:] = None

        if self.any_ovlp():
            if ref is None:
                raise RuntimeError("Cannot read overlaps without a reference Molecule.")
            mol.ovlp_ss = self.read_ovlp(mol.name_a.astype("<U2"), mol.pos_ad, ref.pos_ad)
            mol.adjust_ovlp()
        else:
            mol.ovlp_ss[:] = None

        if self.any_dip():
            mol.dipmom_ssd = self.read_dipmom()
        else:
            mol.dipmom_ssd[:] = None

        HamTransform().transform(mol)
        os.chdir("..")

    def write(self, mol: Molecule):
        self._natoms = mol.n_atoms
        with open(f"{self._file}.xyz", "w") as file:
            file.write(mol.to_xyz())
        self._method(self, mol)

    def backup_wf(self, target: str):
        pass

    def recover_wf(self, source: str):
        pass

    def initiate(self): pass

    @abstractmethod
    def execute(self): pass

    @abstractmethod
    def read_ham(self): pass

    @abstractmethod
    def read_grad(self): pass

    @abstractmethod
    def read_nac(self): pass

    @abstractmethod
    def read_ovlp(self): pass

    @abstractmethod
    def read_dipmom(self): pass