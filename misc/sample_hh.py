import numpy as np
from classes.constants import units

mass = 1/units["amu"]
shift = np.array([-2,0])
nsamp = 1000
freq = 1
qs = np.random.normal(size=(nsamp,2), scale=1/mass/freq)
ps = np.random.normal(size=(nsamp,2), scale=mass*freq)