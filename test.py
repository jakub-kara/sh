from classes.meta import Singleton, Factory, SingletonFactory

class Parent(metaclass = Factory):
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
