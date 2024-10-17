from .composite import CompositeIntegrator
from .rkn import RKN4, RKN8
from .sy import SYAM4
from .vv import VelocityVerlet
from .tdc import *

def select_nucupd(key):
    items = {
        "rkn4": RKN4,
        "rkn8": RKN8,
        "sy4": SYAM4,
        "vv": VelocityVerlet,
    }

    nucupd = CompositeIntegrator()
    nucupd.bind_integrator(items[key](), 0)
    if "rkn" not in key:
        nucupd.bind_integrator(RKN4())
    if "rkn8" not in key:
        nucupd.bind_integrator(RKN8())
    if "sy4" in key:
        nucupd.bind_integrator(RKN4(), -1)

    return nucupd

def select_tdcupd(key):
    items = {
        "hst": HST,
        "hstsharc": HSTSharc,
        "nacme": NACME,
        "npi": NPI,
        "npimeek": NPIMeek,
        "npisharc": NPISharc,
        "ld": LD,
    }
    return items[key]