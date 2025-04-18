import pickle, time
from abc import ABC, abstractmethod
# from functools import partial
from classes.meta import Singleton, Decorator, DecoratorDistributor, Selector, Factory
from classes.molecule import Molecule, MoleculeMixin

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
