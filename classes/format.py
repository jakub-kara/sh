# from enum import Enum
import numpy as np
from .constants import convert

BORDER = "="*40

class Size:
    S = 10
    M = 15
    L = 20

def fmt(val, width = None):
    if width is None:
        width = Size.L

    if np.iscomplex(val):
        form = f"< {width}.{width-8}e"
        return format(val.real, form) + format(val.imag, form)
    elif isinstance(val, int):
        form = f"< {width}.0f"
    elif isinstance(val, float):
        form = f"< {width}.{width-8}e"
    else:
        form = f"<{width}"
    return format(val, form)

def _resolve_args(*args):
    units = []
    width = None
    for arg in args:
        if isinstance(arg, str):
            units.append(arg)
        elif isinstance(arg, int):
            width = arg
    return width, units

def cfmt(val, *args):
    width, units = _resolve_args(*args)
    return fmt(convert(val, *units), width)

def cfmtu(val, *args):
    width, units = _resolve_args(*args)
    if len(units) > 0:
        unit = units[-1]
    return fmt(convert(val, *units), width) + unit
