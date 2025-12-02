from classes.constants import convert

class Validator:
    _type = None

    def __init__(self, *children, op = None):
        self._child: list[Validator] = children
        self._op = op

    def convert(self, val):
        if len(self._child) == 0:
            return self._convert(val)
        else:
            for child in self._child:
                chk, res = child.check_convert(val)
                if chk and self._op == "or":
                    return res
                if not chk and self._op == "and":
                    return ...
            return res

    def _convert(self, val):
        try:
            return self._type(val)
        except:
            return ...

    def check(self, res):
        return res is not ...

    def check_convert(self, vals):
        res = self.convert(vals)
        return self.check(res), res

    def __or__(self, other: "Validator"):
        return Validator(self, other, op = "or")

    def __and__(self, other: "Validator"):
        return Validator(self, other, op = "and")

class AnyValidator(Validator):
    def _convert(self, val):
        return val

    def check(self, res):
        return res is not ...

class BoolValidator(Validator):
    _type = bool
    _true = [True, "true", 1]
    _false = [False, "false", 0]

    def _convert(self, val):
        if isinstance(val, str):
            val = val.lower()
        if val in self._true:
            return True
        if val in self._false:
            return False
        return ...

class IntValidator(Validator):
    _type = int

class FloatValidator(Validator):
    _type = float

class PositiveValidator(Validator):
    _type = "positive"

    def __init__(self, *args, sub = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sub = sub

    def _convert(self, val):
        temp = self._sub.convert(val)
        try:
            return temp if temp >= 0 else ...
        except:
            return ...

    def __call__(self, other: "Validator"):
        return PositiveValidator(*self._child, sub = other, op = self._op)

class StrValidator(Validator):
    _type = str

    def _convert(self, val):
        if val is None or val is ...:
            return ...
        val = str(val)
        if len(val) == 0:
            return ...
        return val

class ListValidator(Validator):
    _type = list
    _sub = None

    def __init__(self, *args, sub = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sub = sub

    def __getitem__(self, key: Validator):
        return ListValidator(*self._child, sub = key, op = self._op)

    def _to_list(self, vals):
        sqz = False
        if not isinstance(vals, (list, tuple, set)):
            sqz = True
            if isinstance(vals, str):
                vals = vals.replace("\n", ",").split(",")
                sqz = len(vals) == 1
            else:
                vals = [vals]
        return vals, sqz

    def _from_list(self, vals, sqz=False):
        if sqz:
            return vals[0]
        return vals

    def _convert(self, vals):
        vals, sqz = self._to_list(vals)
        res = [self._sub.convert(val) for val in vals]
        return self._from_list(res, sqz)

    def check(self, res):
        res, _ = self._to_list(res)
        if len(res) == 0:
            return False
        return all([val is not ... for val in res])

class NoneValidator(Validator):
    _type = None
    _none = [None, "none", "null", ""]

    def _convert(self, val):
        if isinstance(val, str):
            val = val.lower()
        return None if val in self._none else ...

class UnitValidator(Validator):
    _type = "unit"

    def _convert(self, val):
        try:
            return convert(val, "au")
        except:
            return ...
