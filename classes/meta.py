from abc import ABCMeta

# class Factory(ABCMeta):
class Factory(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for sup in cls.mro():
            if type(sup) == type(cls) and hasattr(sup, "_keys"):
                break
        else:
            cls._keys = {}
        cls.__init_subclass__ = classmethod(Factory._initsub)

    def __getitem__(cls, key):
        if key not in cls._keys:
            raise ValueError(f"{key} option not found among the descendents of {cls}.")
        return cls._keys[key]

    def _initsub(cls, key=None, **kwargs):
        if key is not None:
            cls._keys[key] = cls

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

class DynamicClassProxy:
    def __call__(self, kls, kls_name):
        dyncls = getattr(kls, kls_name)
        dyninst = DynamicClassProxy()
        dyninst.__class__ = dyncls
        return dyninst