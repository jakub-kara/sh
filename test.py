import time
from classes.meta import Factory, SingletonFactory, Singleton

class Parent(metaclass = SingletonFactory):
    def __init__(self, x):
        self.x = x

class Child1(Parent, key = 1):
    pass

class Grandchild(Child1, key = 3):
    pass

class Child2(Parent, key = 2):
    pass

x = Parent(x=1, key=3)

breakpoint()
