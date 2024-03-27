import random
import numpy as np
import math

n = 100000
sigma = 1
npr = np.random.default_rng().normal(scale=sigma, size=n)
shr = np.zeros(n)
for i in range(n):
    while True:
        randx = random.random()*10 - 5
        prob = math.exp(-randx**2)
        if prob > random.random():
            break
    shr[i] = randx
breakpoint()