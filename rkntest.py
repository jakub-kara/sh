import numpy as np

subs = 4
a = np.array([
    1/18,
    0,
    2/9,
    1/3,
    0,
    1/6
])
c = np.array([0,    1/3,    2/3,    1])
b = np.array([13/120,   3/10,   3/40,   1/60])
d = np.array([1/8,      3/8,    3/8,    1/8])

# subs = 3
# a = np.array([
#     1/8,
#     0,
#     1/2
# ])
# c = np.array([0, 1/2, 1])
# b = np.array([1/6, 1/3, 0])
# d = np.array([1/6, 2/3, 1/6])

def update(xin: np.ndarray, vin: np.ndarray, ain: np.ndarray, fun, dt: float):
    # helper function for triangular numbers
    def tri(x):
        return x * (x + 1) // 2

    xtmp = np.zeros((subs, *xin.shape))
    atmp = np.zeros((subs, *xin.shape))
    xtmp[0] = xin
    atmp[0] = ain

    # RKN integration substep-by-substep
    for i in range(1, subs):
        # evaluate intermediate position and acceleration
        xtmp[i] = xin + dt * c[i] * vin + dt**2 * np.einsum("j,j...->...", a[tri(i-1):tri(i)], atmp[:i])
        atmp[i] = fun(xtmp[i])

    # find new position and velocity from all the substeps
    xout = xin + dt * vin + dt**2 * np.einsum("j,j...->...", b, atmp)
    aout = fun(xout)
    vout = vin + dt * np.einsum("j,j...->...", d, atmp)

    return xout, vout, aout

def kin(v):
    return 0.5 * np.sum(v**2)

def pot(x):
    return 0.5 * np.sum(x**2)

def func(x):
    return -x

dt = 0.1
tmax = 10
t = 0
x0 = np.array([1])
a0 = func(x0)
v0 = np.array([0])

while t <= tmax:
    print(t, x0[0], v0[0], pot(x0) + kin(v0))
    x0, v0, a0 = update(x0, v0, a0, func, dt)
    t += dt