import numpy as np
import matplotlib.pyplot as plt

from models import tully_1, tully_1m, get_deriv_coupling, get_energies_and_states

n = 1000
x = np.linspace(-10,10,n,endpoint=True)
hdiab = np.zeros((n,2,2))
hdiag = np.zeros((n,2,2))
coup = np.zeros((n))

for i in range(n):
    hdiab[i], gradh = tully_1m(np.array([[x[i],0,0]]))
    hdiag[i], state = get_energies_and_states(hdiab[i])
    coup[i] = get_deriv_coupling(state, gradh, hdiag[i])[0,1,0,0]
    if x[i] > 0:
        coup[i] *= -1

fig, ax = plt.subplots()
ax.plot(x, hdiag[:,0,0])
ax.plot(x, hdiag[:,1,1])
axt = ax.twinx()
axt.plot(x, coup, "g")
plt.show()