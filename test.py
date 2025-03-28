import pickle, time
from abc import ABCMeta, abstractmethod
# from functools import partial
from classes.meta import Singleton, Factory, SingletonFactory
from dynamics.base import Dynamics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bonds", action="append", nargs=2, type=int)
parser.add_argument("-i", "--trajs", action="store", nargs="*")
args = parser.parse_args()

breakpoint()

class Parent(metaclass = Factory):
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

x = Parent[2](x=1)
x.do(1,2)
breakpoint()
