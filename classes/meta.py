from abc import ABC, abstractmethod
import functools

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls.initialised:
            return cls._instances[cls.par]
        else:
            obj = super().__call__(*args, **kwargs)
            cls._instances[cls.par] = obj
            return obj

    @property
    def par(cls):
        return [x for x in cls.mro() if type(x) == type(cls)][-1]

    def reset(cls):
        cls._instances.pop(cls.par, None)

    @property
    def initialised(cls):
        return cls.par in cls._instances.keys()

    @classmethod
    def save(cls):
        return cls._instances

    @classmethod
    def restart(cls, instances: dict):
        cls._instances.update(instances)


class Factory:
    _products = {}

    def __init__(self, base, selector = None):
        self._base = base
        self._selector: Selector = selector
        self._mixins = []

    def add_mixins(self, *args):
        for arg in args:
            if self._selector is None:
                self._mixins.append(arg)
            else:
                self._mixins.append(self._selector.select(arg))
        return self

    def create(self):
        # remove duplicates
        self._mixins = list(set(self._mixins))
        name = self._base.__name__ + "".join(x.__name__ for x in self._mixins)
        self._products[name] = [self._base] + self._mixins
        prod = self._get_product(name)
        setattr(self.__class__, name, prod)
        return prod

    def _get_product(self, name):
        kls = type(name, tuple(self._mixins + [self._base]), {})
        kls.__reduce__ = lambda inst: (DynamicClassProxy(), (Factory, inst.__class__.__name__), inst.__dict__.copy())
        return kls

    @classmethod
    def save(cls):
        return cls._products

    @classmethod
    def restart(cls, dic: dict):
        cls._products = dic
        for name, mixins in dic.items():
            setattr(cls, name, Factory(mixins[0]).add_mixins(*mixins[1:]).create())

class Selector:
    key = None
    _keys = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        if cls._keys is None:
            cls._keys = {}
        if cls.key is not None:
            cls._keys[cls.key] = cls
        # cls._root = Selector in cls.__bases__

    @classmethod
    def select(cls, key):
        return cls._keys[key]

class DynamicClassProxy:
    def __call__(self, kls, kls_name):
        dyncls = getattr(kls, kls_name)
        dyninst = DynamicClassProxy()
        dyninst.__class__ = dyncls
        return dyninst

class DecoratorDistributor:
    def __init_subclass__(cls):
        super().__init_subclass__()
        for base in cls.__bases__:
            for name, attr in base.__dict__.items():
                if not callable(attr):
                    continue

                base_decorators = cls.get_all_decorators(attr)
                if not base_decorators:
                    continue

                child_func = cls.__dict__.get(name)
                if child_func is None:
                    continue

                child_decorators = cls.get_all_decorators(child_func)

                for dec in base_decorators:
                    if not any(isinstance(d, type(dec)) for d in child_decorators):
                        child_func = dec(child_func)

                setattr(cls, name, child_func)

    @staticmethod
    def get_all_decorators(func):
        decorators = []
        while hasattr(func, "decorator"):
            decorators.append(func.decorator)
            func = getattr(func, "func", None)
            if func is None:
                break
        return decorators[::-1]

class Wrapper:
    def __init__(self, func, decorator):
        functools.update_wrapper(self, func)
        self.func = func
        self.decorator: Decorator = decorator

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return functools.partial(self.__call__, instance)

    def __call__(self, instance, *args, **kwargs):
        if self.decorator.id in self.decorator.active:
            return self.func(instance, *args, **kwargs)

        self.decorator.activate()
        res = self.decorator.run(self.func, instance, *args, **kwargs)
        self.decorator.deactivate()
        return res

class Decorator(ABC):
    active = None

    def __init__(self, id):
        self.id = id

    def __init_subclass__(cls):
        cls.active = []

    def __call__(self, func):
        return Wrapper(func, self)

    def activate(self):
        self.active.append(self.id)

    def deactivate(self):
        self.active.remove(self.id)

    @abstractmethod
    def run(self, func, instance, *args, **kwargs):
        pass
