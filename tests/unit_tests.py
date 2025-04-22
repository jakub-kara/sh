import main
import sys, os
import time
import unittest
import tracemalloc, traceback
from copy import deepcopy

default = {
    "dynamics": {
        "name": "model",
        "tmax": 2000,
        "dt": 20,
        "method": "fssh",
        "prob": "tdc",
        "initstate": 1,
        "backup": True
    },
    "nuclear": {
        "input": "geom.xyz",
        "nuc_upd": "vv"
    },
    "quantum": {
        "tdc_upd": "nacme",
        "coeff_upd": "tdc",
        "n_substeps": 50
    },
    "electronic": {
        "program": "model",
        "method": "ho2",
        "states": 2
    },
    "output": {
        "file": "out",
        "record": ["act", "pop", "pes", "pen", "ken", "ten", "nacdt"],
        "timer": ["est", "coe", "tdc", "tot", "sav", "wrt"]
    }
}

# automate
dyns = ["fssh", "mash", "ehr", "mce"]
# dyns = ["fssh"]

class Args:
    def __init__(self):
        self.restart = False
        self.icond = False

class PrintSuppressor:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Test(unittest.TestCase):
    pass

def test_template(inp: dict):
    def test(self):
        orig = os.getcwd()
        main.run_dynamics(Args(), deepcopy(inp))
        os.chdir(orig)
    return test

if __name__ == '__main__':
    for dyn in dyns:
        inp = deepcopy(default)
        inp["dynamics"]["method"] = dyn
        setattr(Test, f"test_{dyn}", test_template(inp))

    unittest.main(verbosity=2, buffer=True)
