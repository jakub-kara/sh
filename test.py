import pickle, time
from abc import ABCMeta
# from classes.meta import Singleton, Factory, SingletonFactory
from dynamics.dynamics import Dynamics

class Factory(ABCMeta):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for sup in cls.mro():
            if type(sup) == type(cls) and hasattr(sup, "_keys"):
                break
        else:
            cls._keys = {}
        cls.__init_subclass__ = classmethod(Factory._initsub)

    @property
    def par(cls):
        return [x for x in cls.mro() if type(x) == type(cls)][-1]

    def __getitem__(cls, key):
        if key not in cls._keys:
            raise ValueError(f"{key} option not found among the descendents of {cls}.")
        return cls._keys[key]

    def _initsub(cls):
        if hasattr(cls, "key"):
            cls._keys[cls.key] = cls

        if cls != cls.par:
            for key, val in cls.par.__dict__.items():
                if hasattr(val, "timer"):
                    setattr(cls, key, val.timer(getattr(cls, key)))

class Timer:
    _timers = []

    def __init__(self, msg, id):
        self._msg = msg
        self._id = id

    def __call__(self, func):
        def inner(*args, **kwargs):
            if self._id in self._timers:
                return func(*args, **kwargs)
            else:
                self._timers.append(self._id)
                t0 = time.time()
                print(self._msg)
                res = func(*args, **kwargs)
                print(f"Time: {time.time() - t0 :.4f}")
                self._timers.remove(self._id)
                return res
        inner.timer = self
        return inner

class Parent(metaclass = Factory):
    a = 1
    def __init__(self, x):
        self.x = x

    @Timer("Parent", id = 1)
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
