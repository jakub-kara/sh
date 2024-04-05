import numpy as np

pos = np.array([
    [-1,  0, -1/np.sqrt(2)],
    [ 1,  0, -1/np.sqrt(2)],
    [ 0, -1,  1/np.sqrt(2)],
    [ 0,  1,  1/np.sqrt(2)]
])

mass = np.ones(pos.shape[0])

inertia = np.zeros((3,3))
for a in range(pos.shape[0]):
    inertia[0,0] += mass[a]*(pos[a,1]**2 + pos[a,2]**2)
    inertia[1,1] += mass[a]*(pos[a,0]**2 + pos[a,2]**2)
    inertia[2,2] += mass[a]*(pos[a,0]**2 + pos[a,1]**2)
    inertia[0,1] -= mass[a] *pos[a,0] * pos[a,1]
    inertia[0,2] -= mass[a] *pos[a,0] * pos[a,2]
    inertia[1,2] -= mass[a] *pos[a,1] * pos[a,2]
inertia[1,0] = inertia[0,1]
inertia[2,0] = inertia[0,2]
inertia[2,1] = inertia[1,2]

in2 = np.einsum("a,aij->ij", mass, np.einsum("ij,al->aij", np.eye(3), pos**2) - np.einsum("ai,aj->aij", pos, pos))
breakpoint()
