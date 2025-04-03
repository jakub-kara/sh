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

def update_vv(xin: np.ndarray, vin: np.ndarray, ain: np.ndarray, fun, dt: float):
    xin = xin[-1]
    vin = vin[-1]
    ain = ain[-1]

    xout = xin + vin*dt + 0.5*ain*dt**2
    aout = fun(xout)
    vout = vin + 0.5*(ain + aout)*dt

    return xout, vout, aout

def kin(v):
    return 0.5 * v**2

def pot(x):
    return 0.5 * x**2

def func(x):
    return -x

dt = 0.1
tmax = 10
t = 3*dt
x = np.array([np.cos(i*dt) for i in range(4)])
a = func(x)
v = np.array([-np.sin(i*dt) for i in range(4)])
# breakpoint()

while t <= tmax:
    print(t)
    print(np.cos(t), x[-1])
    print(-np.sin(t), v[-1])
    print(pot(x[-1]) + kin(v[-1]))
    print()
    xout, vout, aout = update_sy(x, v, a, func, dt)
    x[:-1] = x[1:]
    x[-1] = xout
    v[:-1] = v[1:]
    v[-1] = vout
    a[:-1] = a[1:]
    a[-1] = aout

    t += dt