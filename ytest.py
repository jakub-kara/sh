import numpy as np

w0 = -np.cbrt(2) / (2 - np.cbrt(2))
w1 = 1 / (2 - np.cbrt(2))

c1 = w1/2
c2 = (w0 + w1)/2
c3 = c2
c4 = c1

d1 = w1
d2 = w0
d3 = w1

def tot(x, v):
    return (x**2 + v**2) / 2

def acc(x):
    return -x

def y4(x, v, dt):
    x += c1 * v * dt
    v += d1 * acc(x) * dt

    x += c2 * v * dt
    v += d2 * acc(x) * dt

    x += c3 * v * dt
    v += d3 * acc(x) * dt

    x += c4 * v * dt

    return x, v

def vv(x, v, dt):
    v += acc(x) * dt / 2
    x += v * dt
    v += acc(x) * dt / 2
    return x, v

t = 0
dt = 0.1
x = 0
v = 1
while t <= 10:
    print(tot(x, v), x - np.sin(t))
    x, v = y4(x, v, dt)
    breakpoint()
    t += dt
