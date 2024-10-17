import numpy as np
from classes.bundle import Bundle
from dynamics.sh.fssh import SurfaceHopping

config = {
    "control": {
        "name": "model",
        "tunit": "au",
        "tmax": 50,
        "dt": 0.2,
        "nqsteps": 20,
    },
    "nuclear": {
        "input": "geom.xyz",
        "nucupd": "vv",
    },
    "electronic": {
        "program": "model1d",
        "path": "",
        "type": "sub_1",
        "states": np.array([2,0,0]),
        "options": {
            "basis": "6-31g**",
            "closed": 7, 
            "active": 9,
            "nel": 16,
            "sa": 3,
            "mld": False,
            "df": False
        },
        "tdcupd": "nacme",
        "cupd": "propmat",
    },
    "dynamics": {
        "method": "mce",
        "type": "fssh",
        "decoherence": "edc",
        "initstate": 1,
    },
    "output": {
        "file": "data/out",
        "record": ["active", "pop", "pes", "pen", "ken", "en"],
    }
}

bundle = Bundle(config)

while not bundle.is_finished:
    bundle.run_step()

breakpoint()