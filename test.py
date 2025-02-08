import pickle
import time
from classes.meta import Factory, SingletonFactory, Singleton
from classes.molecule import Molecule, MoleculeFactory

class Parent(metaclass = SingletonFactory):
    def __init__(self, x):
        self.x = x

class Child1(Parent, key = 1):
    pass

class Grandchild(Child1, key = 3):
    pass

class Child2(Parent, key = 2):
    pass

x = Parent[3](x=1)

mol = MoleculeFactory.create_molecule(n_states=1, mixins=("bloch", ))

with open("temp", "wb") as f:
    pickle.dump(mol, f)

with open("temp", "rb") as f:
    temp = pickle.load(f)

breakpoint()
