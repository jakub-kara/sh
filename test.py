import time

class Factory(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for sup in cls.mro():
            if type(sup) == type(cls) and hasattr(sup, "_keys"):
                break
        else:
            cls._keys = {}
        cls.__init_subclass__ = classmethod(Factory._initsub)
        cls.__new__ = classmethod(Factory._new)

    def __call__(cls, *, key=None, **kwargs):
        obj = cls.__new__(key, **kwargs)
        obj.__init__(**kwargs)
        return obj

    def _initsub(cls, key=None, **kwargs):
        if key is not None:
            cls._keys[key] = cls

    def _new(cls, key=None, **kwargs):
        if key in cls._keys:
            sub = cls._keys[key]
        else:
            sub = cls
        return object.__new__(sub)

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_subs"):
            cls._subs = Singleton._allsubs(cls)
            cls._subs = [sub.__name__ for sub in cls._subs]
        inter = list(set(cls._subs) & set(cls._instances))

        if len(inter) == 0:
            obj = super().__call__(*args, **kwargs)
            cls._instances[cls.__name__] = obj
            return obj
        elif len(inter) == 1:
            return cls._instances[inter[0]]
        else:
            raise RuntimeError(f"More than one instance of singleton among {cls} and its descendants.")

    def _allsubs(cls):
        return set([cls]).union(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in Singleton._allsubs(c)])

    def reset(cls):
        subs = Singleton._allsubs(cls)
        for sub in subs:
            cls._instances.pop(sub, None)

class SingletonFactory(Singleton, Factory):
    pass

class Parent(metaclass = SingletonFactory):
    def __init__(self, x):
        self.x = x

class Child1(Parent, key = 1):
    pass

class Grandchild(Child1, key = 3):
    pass

class Child2(Parent, key = 2):
    pass

def func(par: Parent):
    par.x += 1

def gunc():
    par = Parent()
    par.x += 1

par = Parent(key=3, x=0)
t = time.time()
for i in range(10000):
    func(par)
print(time.time() - t)

Parent.reset()

par = Parent(key=3, x=0)
t = time.time()
cls = Parent
for i in range(10000):
    # gunc()
    if not hasattr(cls, "_subs"):
        cls._subs = Singleton._allsubs(cls)
    inter = list(set(cls._subs) & set(cls._instances))
    # cls._instances[inter[0]]
    inter[0]

print(time.time() - t)


breakpoint()
