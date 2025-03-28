from abc import ABCMeta, abstractmethod

class Factory(ABCMeta):
# class Factory(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if cls == cls.par:
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
                if key == "__abstractmethods__" or getattr(cls, key) is val:
                    continue
                if hasattr(val, "decors"):
                    for decor in val.decors:
                        setattr(cls, key, decor(getattr(cls, key)))

class Singleton(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls in cls._instances.keys():
            return cls._instances[cls]
        else:
            obj = super().__call__(*args, **kwargs)
            cls._instances[cls] = obj
            return obj

    def reset(cls):
        cls._instances.pop(cls, None)

    @property
    def initialised(cls):
        return cls in cls._instances

    @staticmethod
    def restart(instances: dict):
        Singleton._instances.update(instances)

    @staticmethod
    def save():
        return Singleton._instances

class SingletonFactory(Singleton, Factory):
    def __call__(cls, *args, **kwargs):
        par = cls.par
        if par in cls._instances.keys():
            return cls._instances[par]
        else:
            obj = super().__call__(*args, **kwargs)
            cls._instances[par] = obj
            return obj

    def reset(cls):
        cls._instances.pop(cls.par, None)

class DynamicClassProxy:
    def __call__(self, kls, kls_name):
        dyncls = getattr(kls, kls_name)
        dyninst = DynamicClassProxy()
        dyninst.__class__ = dyncls
        return dyninst

class Decorator:
    def __init__(self):
        pass

    def __call__(self, func):
        inner = lambda *args, **kwargs: self._inner(func, *args, **kwargs)
        if not hasattr(func, "decors"):
            inner.decors = []
        else:
            inner.decors = func.decors
        inner.decors.append(self)
        return inner

    @abstractmethod
    def _inner(self, func, *args, **kwargs):
        pass