import numpy as np

class sy:
    key = "sy4"
    steps = 4
    a = np.array([1, -1, 0, -1, 1])
    b = np.array([0, 5/4, 1/2, 5/4, 0])


class am:
    name = "am4"
    steps = 4
    order = 4
    b = np.array([1/24, -5/24, 19/24, 3/8])
    c = -19/720

steps = 4

def update_sy(xin: np.ndarray, vin: np.ndarray, ain: np.ndarray, fun, dt: float):
    xin = xin[-steps:]
    vin = vin[-steps:]
    ain = ain[-steps:]

    # find new position as a weighted sum of previous positions and accelerations
    xout = -np.einsum("j,j...->...", sy.a[:-1], xin) + dt**2*np.einsum("j,j...->...", sy.b[:-1], ain)
    xout /= sy.a[-1]

    # calculate new acceleration
    aout = fun(xout)

    # calculate new velocity from new acceleration, previous velocities, and previous accelerations
    vout = vin[-1] + dt*np.einsum("j,j...->...", am.b[:-1], ain[1:]) + dt*am.b[-1]*aout

    return xout, vout, aout

def update_vv(xin: np.ndarray, vin: np.ndarray, ain: np.ndarray, fun, rin: float):
    fact = 1 + eps * ctrl(xin, vin, ain) / rin
    fact = max(min(fact, 2), 0.5)
    rout = rin * fact

    vout = vin + 0.5 * eps * ain / rout
    xout = xin + eps * vout / rout
    aout = fun(xout)
    vout = vout + 0.5 * eps * aout / rout

    return xout, vout, aout, rout

def kin(v):
    return 0.5 * v**2

def pot(x):
    return 0.5 * x**2

def func(x):
    return -x

def ctrl(x, v, a):
    delta = 1e-8
    return alpha * v * a / (v**2 + delta)

def report():
    print(step, t)
    print(np.cos(t), x)
    print(-np.sin(t), v)
    print(pot(x) + kin(v))
    print()

def write():
    file.write(f"{t} {x} {v} {a} {np.abs(pot(x) + kin(v) - 1/2)}\n")

alpha = 1
eps = 0.1
r0 = 1
tmax = 10
step = 0
t = 0
x = 1
a = func(x)
v = 0
r = r0 - eps*ctrl(x, v, a) / 2
file = open("out.dat", "w")

while t <= tmax:
    report()
    write()
    x, v, a, r = update_vv(x, v, a, func, r)
    t += eps / r
    step += 1

report()
write()
file.close()