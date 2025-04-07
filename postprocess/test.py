import numpy as np
from scipy.linalg import expm

nt = 2
alpha = 4.7
pos = lambda t: np.array([-np.cos(t), np.cos(t)])
mom = lambda t: np.array([np.sin(t), -np.sin(t)])
acc = lambda x: -x
pot = lambda x: 1/2*x**2
grd = lambda x: x
kin = lambda p: 1/2*p**2

d = lambda x1, x2: x2 - x1
m = lambda x1, x2: (x1 + x2) / 2

cg_ovl = lambda x1, x2, p1, p2: np.exp(-alpha/2*d(x1,x2)**2 - 1/8/alpha*d(p1,p2)**2 + 1j*(x1*p1 - x2*p2 + m(x1,x2)*d(p1,p2)))
cg_x   = lambda x1, x2, p1, p2: (1j/4/alpha*d(p1,p2) + m(x1,x2)) * cg_ovl(x1, x2, p1, p2)
cg_nab = lambda x1, x2, p1, p2: (1j*m(p1,p2) + alpha*d(x1,x2)) * cg_ovl(x1, x2, p1, p2)
cg_lap = lambda x1, x2, p1, p2: (2j*alpha*d(x1,x2)*m(p1,p2) - alpha + alpha**2*d(x1,x2)**2 - m(p1,p2)**2) * cg_ovl(x1, x2, p1, p2)
cg_dx2 = lambda x1, x2, p1, p2: (-1j*m(p1,p2) - alpha*d(x1,x2)) * cg_ovl(x1, x2, p1, p2)
cg_dp2 = lambda x1, x2, p1, p2: (-1/4/alpha*d(p1,p2) - 1j/2*d(x1,x2)) * cg_ovl(x1, x2, p1, p2)

def cg_pot(x1, x2, p1, p2):
    temp = cg_ovl(x1, x2, p1, p2) * (pot(x1) + pot(x2))/2
    temp += grd(x1) * (cg_x(x1, x2, p1, p2) - x1*cg_ovl(x1, x2, p1, p2)) / 2
    temp += grd(x2) * (cg_x(x1, x2, p1, p2) - x2*cg_ovl(x1, x2, p1, p2)) / 2
    return temp

tbf_ovl = lambda x1, x2, p1, p2: cg_ovl(x1, x2, p1, p2)

def tbf_ham(x1, x2, p1, p2):
    return -1/2*cg_lap(x1, x2, p1, p2) + cg_pot(x1, x2, p1, p2)

def tbf_dt(x1, x2, p1, p2):
    temp = p2*cg_dx2(x1, x2, p1, p2) + acc(x2)*cg_dp2(x1, x2, p1, p2) + 1j/2*p2**2*cg_ovl(x1, x2, p1, p2)
    temp += cg_ovl(x1, x2, p1, p2) * (-1j*pot(x2))
    return temp

def set_vars(t):
    return *pos(t), *mom(t)

def to_mat(fn, x1, x2, p1, p2):
    return np.array([
        [fn(x1,x1,p1,p1), fn(x1,x2,p1,p2)],
        [fn(x2,x1,p2,p1), fn(x2,x2,p2,p2)]
    ])

x1, x2, p1, p2 = set_vars(0)
ovl = to_mat(tbf_ovl, x1, x2, p1, p2)
coeff = np.ones(2, dtype=np.complex128)
coeff /= np.sqrt(np.einsum("a, ab, b ->", coeff.conj(), ovl, coeff))
t = 0
dt = 1e-3
tmax = 1e-2
while t <= tmax:
    x1, x2, p1, p2 = set_vars(t)
    ham = to_mat(tbf_ham, x1, x2, p1, p2)
    ddt = to_mat(tbf_dt, x1, x2, p1, p2)
    ovl = to_mat(tbf_ovl, x1, x2, p1, p2)
    coeff = expm(-np.linalg.inv(ovl) @ (1j*ham+ddt) * dt) @ coeff
    print(coeff)
    print(np.real(np.einsum("a, ab, b ->", coeff.conj(), ovl, coeff)))
    t += dt

breakpoint()