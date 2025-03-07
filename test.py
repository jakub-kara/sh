import pickle
from classes.meta import Singleton, Factory, SingletonFactory
from dynamics.dynamics import Dynamics
from electronic.electronic import ESTProgram

class Parent(metaclass = Factory):
    a = 1
    def __init__(self, x):
        self.x = x

class Child1(Parent):
    key = 1
    pass

class Grandchild(Child1):
    key = 2
    pass

class Child2(Parent):
    key = 3
    pass

x = Parent[2](x=1)
breakpoint()
