import pickle, time
from abc import ABC, abstractmethod
# from functools import partial
from classes.meta import Singleton, Decorator, DecoratorDistributor, Selector, Factory
from classes.molecule import Molecule, MoleculeMixin

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

arr = np.zeros((2,2), dtype=np.complex128)
arr[0,1] = 1j
arr[1,0] = -1j
# vec = np.arange(2)
vec = np.ones(2) / np.sqrt(2)

for num in np.exp(np.linspace(-6, 0, 20)):
    res = expm(1j*arr*num) @ vec
    print(num)
    print(res)
    print(np.arccos(np.abs(np.vdot(res, vec))))
    print(np.abs(np.einsum("i,j,ij->", vec, vec, arr*num)))
    print()
breakpoint()

molfac = Factory(Molecule, MoleculeMixin)
molfac.add_mixins("sh")
molcls = molfac.create()
mol = molcls(initstate = 0)

with open("temp.pkl", "wb") as pkl:
    pickle.dump(mol, pkl)
breakpoint()


class Parent(Selector, DecoratorDistributor, metaclass = Singleton):
    a = 1
    def __init__(self, x):
        self.x = x

    def do(self, *args, **kwargs):
        print("in Parent")
        print(args)
        print(kwargs)

class Child1(Parent):
    key = 1

    def do(self, *args, **kwargs):
        print("in Child1")
        super().do(*args, **kwargs)

class Grandchild(Child1):
    key = 2

    def do(self, *args, **kwargs):
        print("in Grandchild")
        super().do(*args, **kwargs)

class Child2(Parent):
    key = 3

    def do(self, *args, **kwargs):
        print("in Child2")
        super().do(*args, **kwargs)

x = Parent(10)
x.do(1,2)

temp = pickle.dumps(x)
pickle.loads(temp)
breakpoint()
