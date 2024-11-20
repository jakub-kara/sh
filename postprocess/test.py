import numpy as np
from scipy.linalg import expm

def s(t):
    res = np.eye(2)
    res[0,1] = np.exp(-2*(1-t)**2 - 1/2)
    res[1,0] = res[0,1]
    return res

def h(t):
    res = np.eye(2)
    res[0,1] = (2*(1-t)**2 - 1/2) * np.exp(-2*(1-t)**2 - 1/2)
    res[1,0] = res[0,1]
    return res

def k(t):
    res = np.zeros((2,2), dtype=np.complex128)
    res[0,0] = -1j/2
    res[1,1] = -1j/2
    res[0,1] = (2*(1-t) + 1j/2) * np.exp(-2*(1-t)**2 - 1/2)
    res[1,0] = res[0,1]
    return res

def f(y, s, h, k):
    return -np.linalg.inv(s) @ (1j*h + k) @ y

w0 = np.sqrt(1/(2 * (1 + s(0)[0,1])))
c = np.array([w0, w0], dtype=np.complex128)
dt = 0.01
t = 0
print(k(0))
print(h(0))
print(s(0))
breakpoint()

while t < 1.1:
    print(t)
    print(c)
    print(np.einsum("i,ij,j->", c.conj(), s(t), c))
    print()
    sini = s(t)
    sfin = s(t+dt)
    hini = h(t)
    hfin = h(t+dt)
    kini = k(t)
    kfin = k(t+dt)

    k1 = f(c, sini, hini, kini)
    k2 = f(c + dt*k1/2, (sini+sfin)/2, (hini+hfin)/2, (kini+kfin)/2)
    k3 = f(c + dt*k2/2, (sini+sfin)/2, (hini+hfin)/2, (kini+kfin)/2)
    k4 = f(c + dt*k3, sfin, hfin, kfin)

    # k1 = f(t, c)
    # k2 = f(t + dt/2, c + dt*k1/2)
    # k3 = f(t + dt/2, c + dt*k2/2)
    # k4 = f(t + dt, c + dt*k3)
    c += dt*(k1+2*k2+2*k3+k4)/6
    
    t += dt