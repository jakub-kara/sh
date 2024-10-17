from .models import Model1D
from .molpro import Molpro

def select_est(key):
    items = {
        "model1d": Model1D,
        "molpro": Molpro,
    }
    return items[key]