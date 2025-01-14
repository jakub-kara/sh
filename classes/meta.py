from abc import ABCMeta

class Factory(ABCMeta):
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
        subs = Singleton._allsubs(cls)
        inter = list(set(subs) & set(cls._instances))

        if len(inter) == 0:
            obj = super().__call__(*args, **kwargs)
            cls._instances[cls] = obj
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

    @staticmethod
    def restore(instances: dict):
        Singleton._instances.update(instances)

    @staticmethod
    def save():
        return Singleton._instances

class SingletonFactory(Singleton, Factory):
    pass