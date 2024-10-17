from .sh.fssh import FSSH
from .sh.mash import unSMASH
from .ehr.ehr import SimpleEhrenfest
from .ehr.mce import MultiEhrenfest

def select_dynamics(dyntype: str):
    dyntypes = {
        "fssh": FSSH,
        "mash": unSMASH,
        "ehr": SimpleEhrenfest,
        "mce": MultiEhrenfest,
    }
    return dyntypes[dyntype]