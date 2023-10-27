import numpy as np

def adapt_timestep(self, overlap_ss):
    return np.any(np.diag(overlap_ss) < 0.999)