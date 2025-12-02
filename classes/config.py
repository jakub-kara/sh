from .validator import *

def get_cfg(obj, keys, cfg = None):
    if not isinstance(obj, type):
        obj = obj.__class__
    return Config.get(obj, keys, cfg)

class Config():
    # saves all configs
    dic: dict[str, "Config"] = {}
    # classes dependent on config options
    sel: dict[type, "Config"] = {}
    comp: dict = {}
    _waiting = {}

    DYNAMICS = "dynamics"
    NUCLEAR = "nuclear"
    QUANTUM = "quantum"
    ELECTRONIC = "electronic"
    OUTPUT = "output"

    BOOL = BoolValidator()
    INT = IntValidator()
    FLOAT = FloatValidator()
    POSITIVE = PositiveValidator()
    STR = StrValidator()
    UNIT = UnitValidator()
    LIST = ListValidator()
    NONE = NoneValidator()
    ANY = AnyValidator()

    # Ellipsis is used as a placeholder "no default" value, in case None is a valid input
    def __init__(self, key, *, root = False, default = ..., options = None, fix = ..., path: str = None, valid = None):
        self.root = root
        self.key = key
        self.dfl = default
        self.fix = fix
        self.path = path

        if valid is None:
            valid = Config.ANY
        self.valid = valid

        if options is None:
            options = {}
        self.opt = options

        if self.valid is Config.BOOL:
            self.opt = [True, False]

    def __call__(self, kls):
        self.kls = kls
        kls_key = self.get_key(kls, self.key)
        self.dic[kls_key] = self

        if kls_key in self._waiting:
            for opt, (obj, base) in self._waiting[kls_key].items():
                self.add_option(opt, self.key, obj, base)
            self._waiting.pop(kls_key)
        return kls

    def __repr__(self):
        return f"Config of {self.get_key(self.kls, self.key)}"

    @property
    def is_selector(self):
        return len(self.opt) > 0

    def get_val(self, cfg, key = ...):
        if key is ...:
            key = self.key
        if key in cfg:
            return cfg[key]
        elif self.path is not None:
            for k in self.path.split("."):
                if k in cfg:
                    cfg = cfg[k]
            if key in cfg:
                return cfg[key]
        return ...

    def get_res(self, cfg: dict = None, key = ""):
        if cfg is None:
            return self.opt.get(key, self.dfl)

        if self.fix is not ...:
            if key in cfg and cfg[key] != self.fix:
                print(f"Warning: key *{key}* with value *{cfg[key]}* detected in config. However, a mandatory value is *{self.fix}*.")
            return self.fix

        temp = self.get_val(cfg, key)
        if temp is ...:
            temp = self.dfl

        if temp is ...:
            raise ValueError(f"Unspecified key *{key}* in class *{self.kls.__qualname__}* without default value.")

        if isinstance(self.opt, dict) and len(self.opt) > 0:
            return self.opt[temp]

        return temp

    @classmethod
    def get_key(cls, kls, key = ""):
        if isinstance(kls, str):
            return kls + "." + str(key)
        else:
            return kls.__qualname__ + "." + str(key)

    @classmethod
    def match_key(cls, kls: type, key: str):
        for par in kls.mro():
            kls_key = cls.get_key(par, key)
            if kls_key in cls.dic:
                return kls_key
            kls_key = cls.get_key(par)
            if kls_key in cls.dic:
                return kls_key

        raise LookupError(f"Unconfigured key *{key}* for class *{kls.__qualname__}*.")

    @classmethod
    def get_inst(cls, kls, key):
        kls_key = cls.match_key(kls, key)
        return cls.dic[kls_key]

    @classmethod
    def get(cls, kls, keys, cfg: dict):
        sqz = False
        res = []
        if not isinstance(keys, (list, tuple)):
            sqz = True
            keys = [keys]

        for key in keys:
            inst = cls.get_inst(kls, key)
            res.append(inst.get_res(cfg, key))

        if sqz:
            return res[0]
        return res

    @classmethod
    def add_option(cls, opt, key, kls, base = None):
        if base is not None:
            kls_key = cls.get_key(base, key)
        else:
            kls_key = cls.match_key(kls, key)

        if kls_key in cls.dic:
            inst = cls.dic[kls_key]
            inst.opt[opt] = kls
            inst.root = True
            cls.sel[kls] = inst
        else:
            if kls_key not in cls._waiting:
                cls._waiting[kls_key] = {}
            cls._waiting[kls_key][opt] = (kls, base)

    class Option:
        def __init__(self, opt, key, base = None):
            self.opt = opt
            self.key = key
            self.base = base

        def __call__(self, kls):
            Config.add_option(self.opt, self.key, kls, self.base)
            return kls

    class Init:
        THIS = ...
        def __init__(self, *comp):
            self.comp = comp

        def __call__(self, kls):
            for c in self.comp:
                if c is Config.Init.THIS:
                    Config.comp[kls] = kls
                else:
                    Config.comp[c] = kls
            return kls