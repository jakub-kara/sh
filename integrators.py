import numpy as np    
from dataclasses import dataclass
from typing import Callable, Any
from scipy.linalg import expm

from hopping import get_hopping_prob_ddr, check_hop, get_hopping_prob_LD
from classes import Trajectory
from constants import Constants

@dataclass
class RK:
    name: str
    m: int
    n: int
    r: int
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

@dataclass
class RKN:
    name: str = None
    m: int = None
    n: int = None
    r: int = None
    a: np.ndarray = None
    b: np.ndarray = None
    c: np.ndarray = None
    d: np.ndarray = None

@dataclass
class AB:
    name: str = None
    m: int = None
    n: int = None
    r: int = None
    v: Any = None
    a: np.ndarray = None
    b: np.ndarray = None
    c: float = None

@dataclass
class VV:
    name: str
    m: int
    n: int
    r: int

SV = VV(
    name = "vv",
    m = 1,
    n = 1,
    r = 1
)

OV = VV(
    name = "ov",
    m = 1,
    n = 2,
    r = 1
)

# aka explicit Euler
RK1 = RK(
    name = "rk1",
    m = 1,
    n = 1,
    r = 1,
    a = np.zeros((1,1)),
    b = np.ones(1),
    c = np.zeros(1)
)

# aka trapezoidal rule
RK2a = RK(
    name = "rk2a",
    m = 1,
    n = 2,
    r = 2,
    a = np.array([
        1
    ]),
    b = np.array([1/2,   1/2]),
    c = np.array([0,     1])
)

# aka midpoint rule
RK2b = RK(
    name = "rk2b",
    m = 1,
    n = 2,
    r = 2,
    a = np.array([
        1/2
    ]),
    b = np.array([0,     1]),
    c = np.array([0,     1/2])
)

RK3a = RK(
    name = "rk3a",
    m = 1,
    n = 3,
    r = 3,
    a = np.array([
        2/3,
        1/3,
        1/3
    ]),
    b = np.array([1/4,  0,   3/4]),
    c = np.array([0,    2/3, 2/3])
)

RK3b = RK(
    name = "rk3b",
    m = 1,
    n = 3,
    r = 3,
    a = np.array([
        1/2,
        -1,
        2
    ]),
    b = np.array([1/6,  2/3,    1/6]),
    c = np.array([0,    1/2,    1])
)

# aka RK4
RK4a = RK(
    name = "rk4a",
    m = 1,
    n = 4,
    r = 4,
    a = np.array([
        1/2,
        0,
        1/2,
        0,
        0,
        1
    ]),
    b = np.array(
        [1/6,   1/3,    1/3,    1/6]
    ),
    c = np.array(
        [0,     1/2,    1/2,    1]
    )
)

RK4b = RK(
    name = "rk4b",
    m = 1,
    n = 4,
    r = 4,
    a = np.array([
        1/4,
        0,
        1/2,
        1,
        -2,
        2
    ]),
    b = np.array([1/6,  0,   2/3,   1/6]),
    c = np.array([0,    1/4, 1/2,   1])
)

RK4c = RK(
    name = "rk4c",
    m = 1,
    n = 4,
    r = 4,
    a = np.array([
        1/3,
        -1/3,
        1,
        1,
        -1,
        1
    ]),
    b = np.array([1/8,  3/8,    3/8,    1/8]),
    c = np.array([0,    1/3,    2/3,    1])
)

RK5 = RK(
    name = "rk5",
    m = 1,
    n = 6,
    r = 5,
    a = np.array([
        1/4,
        1/8,
        1/8,
        0,
        0,
        1/2,
        3/16,
        -3/8,
        3/8,
        9/16,
        -3/7,
        8/7,
        6/7,
        -12/7,
        8/7
    ]),
    b = np.array([7/90,  0,   32/90,  12/90,  32/90,  7/90]),
    c = np.array([0,     1/4, 1/4,    1/2,    3/4,    1])
)

RK6 = RK(
    name = "rk6",
    m = 1,
    n = 7,
    r = 6,
    a = np.array([
        1/3,
        0,
        2/3,   
        1/12,
        1/3,
        -1/12,   
        25/48,
        -55/24,
        35/48,
        15/8,   
        3/20,
        -11/24,
        -1/8,
        1/2,
        1/10,   
        -261/260,
        33/13,
        43/156,
        -118/39,
        32/195,
        80/39
    ]),
    b = np.array([13/200,   0,  11/40,  11/40,  4/25,   4/25,   13/200]),
    c = np.array([0,        1/3,2/3,    1/3,    5/6,    1/6,    1])
)

RKN4 = RKN(
    name = "rkn4",
    m = 1,
    n = 4,
    r = 4,
    a = np.array([
        1/18,
        0,
        2/9,
        1/3,
        0,
        1/6
    ]),
    c = np.array([0,    1/3,    2/3,    1]),
    b = np.array([13/120,   3/10,   3/40,   1/60]),
    d = np.array([1/8,      3/8,    3/8,    1/8])
)

RKN6 = RKN(
    name = "rkn6",
    m = 1,
    n = 7,
    r = 6,
    a = np.array([
        1/200,
        1/150,
        1/75,
        2/75,
        0,
        4/75,
        9/200,
        0,
        9/100,
        9/200,
        199/3600,
        -19/150,
        47/120,
        -119/1200,
        89/900,
        -179/1824,
        17/38,
        0,
        -37/152,
        219/456,
        -157/1824
    ]),
    c = np.array([0,    1/10,   1/5,    2/5,    3/5,    4/5,    1]),
    b = np.array([61/1008,  0,  475/2016,   25/504, 125/1008, 25/1008, 11/2016]),
    d = np.array([19/288,   0,  25/96,      25/144, 25/144,   25/96,   19/288])
)

RKN8 = RKN(
    name = "rkn8",
    m = 1,
    n = 11,
    r = 8,
    a = np.array([
        49/12800,

        49/9600,
        49/4800,

        16825/381024,
        -625/11907,
        18125/190512,

        23/840,
        0,
        50/609,
        9/580,

        533/68040,
        0,
        5050/641277,
        -19/5220,
        23/12636,

        -4469/85050,
        0,
        -2384000/641277,
        3896/19575,
        -1451/15795,
        502/135,

        694/10125,
        0,
        0,
        -5504/10125,
        424/2025,
        -104/2025,
        364/675,

        30203/691200,
        0,
        0,
        0,
        9797/172800,
        79391/518400,
        20609/345600,
        70609/2073600,

        1040381917/14863564800,
        0,
        548042275/109444608,
        242737/5345280,
        569927617/6900940800,
        -2559686731/530841600,
        -127250389/353894400,
        -53056229/2123366400,
        23/5120,

        -33213637/179088000,
        0,
        604400/324597,
        63826/445875,
        0,
        -6399863/2558400,
        110723/511680,
        559511/35817600,
        372449/7675200,
        756604/839475
    ]),
    c = np.array([0, 7/80, 7/40, 5/12, 1/2, 1/6, 1/3, 2/3, 5/6, 1/12, 1]),
    b = np.array([121/4200, 0, 0, 0, 43/525, 33/350, 17/140, 3/56, 31/1050, 512/5775, 1/550]),
    d = np.array([41/840, 0, 0, 0, 34/105, 9/35, 9/280, 9/280, 9/35, 0, 41/840])
)

AB2 = AB(
    name = "ab2",
    m = 2,
    n = 1,
    r = 2,
    b = np.array([-1/2, 3/2]),
    c = -1/2
)

AB3 = AB(
    name = "ab3",
    m = 3,
    n = 1,
    r = 3,
    b = np.array([5/12, -4/3, 23/12]),    
    c = 5/12
)

AB4 = AB(
    name = "ab4",
    m = 4,
    n = 1,
    r = 4,
    b = np.array([-3/8, 37/24, -59/24, 55/24]),
    c = -3/8
)

AB5 = AB(
    name = "ab5",
    m = 5,
    n = 1,
    r = 5,
    b = np.array([251/720, -637/360, 109/30, -1387/360, 1901/720]),
    c = 251/720
)

AB6 = AB(
    name = "ab6",
    m = 6,
    n = 1,
    r = 6,
    b = np.array([-95/288, 959/480, -3649/720, 4991/720, -2641/480, 4277/1440]),
    c = 19087/60480
)

AB7 = AB(
    name = "ab7",
    m = 7,
    n = 1,
    r = 7,
    b = np.array([19087/60480, -5603/2520, 135713/20160, -10754/945, 235183/20160, -18637/2520, 198721/60480]),
    c = -5257/17280
)

AB8 = AB(
    name = "ab8",
    m = 8,
    n = 1,
    r = 8,
    b = np.array([-5257/17280, 32863/13440, -115747/13440, 2102243/120960, -296053/13440, 242653/13440, -1152169/120960, 16083/4480]),
    c = 1070017/3628800
)

AM2 = AB(
    name = "am2",
    m = 2,
    n = 1,
    r = 2,
    b = np.array([1/2, 1/2]),
    c = -1/12
)

AM3 = AB(
    name = "am3",
    m = 3,
    n = 1,
    r = 3,
    b = np.array([-1/12, 2/3, 5/12]),
    c = 1/24
)

AM4 = AB(
    name = "am4",
    m = 4,
    n = 1,
    r = 4,
    b = np.array([1/24, -5/24, 19/24, 3/8]),
    c = -19/720
)

AM5 = AB(
    name = "am5",
    m = 5,
    n = 1,
    r = 5,
    b = np.array([-19/720, 53/360, -11/30, 323/360, 251/720]),
    c = 3/160
)

AM6 = AB(
    name = "am6",
    m = 6,
    n = 1,
    r = 6,
    b = np.array([3/160, -173/1440, 241/720, -133/240, 1427/1440, 95/288]),
    c = -863/60480
)

AM7 = AB(
    name = "am7",
    m = 7,
    n = 1,
    r = 7,
    b = np.array([-863/60480, 263/2520, -6737/20160, 586/945, -15487/20160, 2713/2520, 19087/60480]),
    c = 275/24192
)

AM8 = AB(
    name = "am8",
    m = 8,
    n = 1,
    r = 8,
    b = np.array([275/24192, -11351/120960, 1537/4480, -88547/120960, 123133/120960, -4511/4480, 139849/120960, 5257/17280]),
    c = -33953/3628800
)

SY2 = AB(
    name = "sy2",
    m = 2,
    n = 1,
    r = 2,
    v = AM2,
    a = np.array([1, -2, 1]),
    b = np.array([0, 1, 0]),
    c = 1/12
)

SY4 = AB(
    name = "sy4",
    m = 4,
    n = 1,
    r = 4,
    v = AM4,
    a = np.array([1, -1, 0, -1, 1]),
    b = np.array([0, 5/4, 1/2, 5/4, 0])
)

SY6 = AB(
    name = "sy6",
    m = 6,
    n = 1,
    r = 6,
    v = AM6,
    a = np.array([1, -2, 2, -2, 2, -2, 1]),
    b = np.array([0, 317/240, -31/30, 291/120, -31/30, 317/240, 0]),
    c = 275/4032
)

SY8 = AB(
    name = "sy8",
    m = 8,
    n = 1,
    r = 8,
    v = AM8,
    a = np.array([1, -2, 2, -1, 0, -1, 2, -2, 1]),
    b = np.array([0, 17671, -23622, 61449, -50516, 61449, -23622, 17671, 0])/12096
)

SY8b = AB(
    name = "sy8b",
    m = 8,
    n = 1,
    r = 8,
    v = AM8,
    a = np.array([1, 0, 0, -1/2, -1, -1/2, 0, 0, 1]),
    b = np.array([0, 192481, 6582, 816783, -156812, 816783, 6582, 192481, 0])/120960
)

SY8c = AB(
    name = "sy8c",
    m = 8,
    n = 1,
    r = 8,
    v = AM8,
    a = np.array([1, -1, 0, 0, 0, 0, 0, -1, 1]),
    b = np.array([0, 13207, -8934, 42873, -33812, 42873, -8934, 13207, 0])/8640
)

def est_wrapper(x: np.ndarray, substep: int, traj: Trajectory):
    traj.est.nacs_setter(traj, substep==0 and traj.est.tdc_updater=='nacme')
    traj.ctrl.substep = substep
    temp = traj.geo.position_mnad[-1,0]
    traj.geo.position_mnad[-1,0] = x
    traj.est.run(traj)
    traj.geo.position_mnad[-1,0] = temp
    traj.geo.force_updater(traj)
    return traj.geo.force_mnad[-1,0]

def ARKN3Solver(y0: np.ndarray, v0: np.ndarray, f0: np.ndarray, func: Callable, fargs: tuple, dt: float, scheme: RKN, *args):
    a1 = 1/2
    b1 = 2/3
    b_1 = -1/3
    b2 = 5/6
    b2_ = 5/12
    b_ = a1*b2
    
    k1 = f0[-1]
    k2 = func(y0[-1] + dt*a1*k1, False, *fargs)
    k_1 = f0[-2]
    k_2 = func(y0[-2] + dt*a1*k_1, False, *fargs)
    
    y1 = y0[-1] + 3/2*dt*v0[-1] + 1/2*dt*v0[-2] + dt**2*b_*(k1 - k_1)
    v1 = v0[-1] + dt*(b1*k1 - b_1*k_1 + b2*(k2 - k_2))
    f1 = func(y1, True, *fargs)
    return y1, v1, f1

def RKNSolver(y0: np.ndarray, v0: np.ndarray, f0: np.ndarray, func: Callable, fargs: tuple, dt: float, scheme: RKN, *args):
    def tri(x):
        return int(x*(x+1)/2)
    
    Y = np.zeros((scheme.n, *y0.shape))
    Y[0] = y0
    F = np.zeros((scheme.n, *y0.shape))
    F[0] = f0
    
    for i in range(1,scheme.n):
        Y[i] = Y[0] + dt*scheme.c[i]*v0 + dt**2*np.einsum("j,j...->...", scheme.a[tri(i-1):tri(i)], F[:i])
        F[i] = func(Y[i], i, *fargs)

    y1 = y0 + dt*v0 + dt**2*np.einsum("j,j...->...", scheme.b, F)
    v1 = v0 + dt*np.einsum("j,j...->...", scheme.d, F)
    f1 = func(y1, 0, *fargs)
    
    return y1, v1, f1

def VVSolver(y0: np.ndarray, v0: np.ndarray, f0: np.ndarray, func: Callable, fargs: tuple, dt: float, *args):
    y1 = y0 + dt*v0 + 0.5*dt**2*f0[-1]
    f1 = func(y1, 0, *fargs)
    v1 = v0 + 0.5*dt*(f0[-1] + f1)
    return y1, v1, f1

def OVSolver(y0: np.ndarray, v0: np.ndarray, f0: np.ndarray, func: Callable, fargs: tuple, dt: float, *args):
    # https://doi.org/10.1103/PhysRevE.65.056706

    #zeta = 1/2 - 1/12*(2*np.sqrt(326) + 36)**(1/3) + 1/(6*(2*np.sqrt(326) + 36)**(1/3))

    y1 = y0 + dt*Constants.zeta*v0
    f1 = func(y1, False, *fargs)
    v1 = v0 + 0.5*dt*f1

    y1 += dt*(1 - 2*Constants.zeta)*v1
    f1 = func(y1, False, *fargs)
    v1 += 0.5*dt*f1

    y1 += dt*Constants.zeta*v1

    f1 = func(y1, True, *fargs)
    return y1, v1, f1

def calculate_am4_coeffs(h0, h1, h2):
    b0 = h2**2*(2*h1+h2)/(h0*(h0+h1)*(h0+h1+h2))/12
    b1 = -h2**2*(2*h0+2*h1+h2)/(h0*h1*(h1+h2))/12
    b2 = (3*h1*(h0+h1) + (h1+h2)*(h0+h1+h2) + h1*(h0+h1+h2) + (h0+h1)*(h1+h2))/(h1*(h0+h1))/12
    b3 = (3*(h1+h2)*(h0+h1+h2) + h1*(h0+h1) + (h1+h2)*(h0+h1) + h1*(h0+h1+h2))/((h1+h2)*(h0+h1+h2))/12

    return AB(
        name = "am4v",
        m = 4,
        r = 4,
        b = np.array([b0,b1,b2,b3]),
        c = None
    )

def am4temp(h0, h1, h2):
    t1 = h0
    t2 = h0+h1
    t3 = h0+h1+h2

    b0 = -(t2-t3)**3 * (2*t1-t2-t3)/((-t1)*(-t2)*(-t3))/12
    b1 = (t2-t3)**3 * (t2+t3)/(t1*(t1-t2)*(t1-t3))/12
    b2 = -(t3-t2)**2 * (t3**2 + 2*t2*t3 + 3*t2**2 - 2*t1*(t3+2*t2))/(t2*(t2-t1)*(t2-t3))/12
    b3 = (t2-t3)**2 * (t2**2 + 2*t2*t3 + 3*t3**2 - 2*t1*(t2+2*t3))/(t3*(t3-t1)*(t3-t2))/12

    return AB(
        name = "am4v",
        m = 4,
        r = 4,
        b = np.array([b3,b2,b1,b0]),
        c = None
    )

def calculate_sy4_coeffs(h0, h1, h2, h3):
    def T(h3, h2, h1, h0):
        return 2*h0*h3*(bt[1]*(h0-2*h1-h2-h3) + bt[2]*(h0+h1-h2-h3) + bt[3]*(h0+h1+2*h2-h3))
    
    def C(h3, h2, h1, h0):
        return 0.5*T(h3,h2,h1,h0) + SY4.a[1]*3*h0*np.sqrt(h1*h2)*h3
    
    bt = SY4.b
    a1 = C(h3,h2,h1,h0)/(h0*h1*(h1+h2+h3))
    a3 = C(h0,h1,h2,h3)/(h3*h2*(h0+h1+h2))
    
    a2 = - (2*h0*h3*(bt[1]+bt[2]+bt[3]) + a1*h0*(h1+h2+h3) + a3*(h0+h1+h2)*h3)
    a2 /= (h0+h1)*(h2+h3)

    a4 = - (a1*h0 + a2*(h0+h1) + a3*(h0+h1+h2))
    a4 /= h0+h1+h2+h3

    a0 = - (a1+a2+a3+a4)

    v = calculate_am4_coeffs(h1, h2, h3)

    return AB(
        name = "sy4v",
        m = 4,
        n = 1,
        r = 4,
        v = v,
        a = np.array([a0,a1,a2,a3,a4]),
        b = h0/h3*SY4.b[:],
        c = None
    )

def AMSolver(y0: np.ndarray, f0: np.ndarray, dt: float, scheme: AB, *args):
    y1 = y0[-1] + dt*np.einsum("j,j...->...", scheme.b, f0)
    return y1

def SYSolver(y0: np.ndarray, v0: np.ndarray, f0: np.ndarray, func: Callable, fargs: tuple, dt: float, scheme: AB, *args):
    y1 = -np.einsum("j,j...->...", scheme.a[:-1], y0) + dt**2*np.einsum("j,j...->...", scheme.b[:-1], f0)
    y1 /= scheme.a[-1]
    f1 = func(y1, 0, *fargs)
    v1 = v0[-1] + dt*np.einsum("j,j...->...", scheme.v.b[:-1], f0[1:]) + dt*scheme.v.b[-1]*f1

    return y1, v1, f1

def shift_values(*args):
    for arr in args:
        for m in range(1, arr.shape[0]):
            arr[m-1] = arr[m]
        arr[m] = np.nan

def interpolate(x: np.ndarray, y: np.ndarray, inp: float):
    poly = np.ones_like(x, dtype=float)
    for j in range(x.shape[0]):
        for i in range(x.shape[0]):
            if (x[i] == x[j]): continue
            poly[j] *= (inp - x[i])/(x[j] - x[i])
    return np.einsum("j...,j->...", y, poly)

def update_force_sh(traj: Trajectory):
    traj.geo.force_mnad[-1,0,:,:] = -traj.pes.nac_ddr_mnssad[-1,0,traj.hop.active, traj.hop.active,:,:]/traj.geo.mass_a[:,None]

def update_force_mfe(traj: Trajectory):
    for a in range(traj.par.n_atoms):
        traj.geo.force_mnad[-1,0,a] = 0
        for s1 in range(traj.par.n_states):
            traj.geo.force_mnad[-1,0,a] -= np.abs(traj.est.coeff_mns[-1,0,s1])**2*traj.pes.nac_ddr_mnssad[-1,0,s1,s1,a]
            for s2 in range(s1):
                traj.geo.force_mnad[-1,0,a] -= 2*np.real(np.conj(traj.est.coeff_mns[-1,0,s1])*traj.est.coeff_mns[-1,0,s2])* \
                    traj.pes.nac_ddr_mnssad[-1,0,s1,s2,a]*(traj.pes.ham_diag_mnss[-1,0,s2,s2] - traj.pes.ham_diag_mnss[-1,0,s1,s1])

    traj.geo.force_mnad[-1,0,:,:] /= traj.geo.mass_a[:,None]
def update_tdc(traj: Trajectory):
    # create time-derivative coupling from position-derivative coupling. Only used when tdc_updater == nacme
    
    traj.pes.nac_ddt_mnss[-1,0] = 0
    for s1 in range(traj.par.n_states):
        for s2 in range(s1):
            traj.pes.nac_ddt_mnss[-1,0,s1,s2] = np.sum(traj.geo.velocity_mnad[-1,0] * traj.pes.nac_ddr_mnssad[-1,0,s1,s2])
            traj.pes.nac_ddt_mnss[-1,0,s2,s1] = -traj.pes.nac_ddt_mnss[-1,0,s1,s2] 

def interpolate(x: np.ndarray, y: np.ndarray, inp: float):
    poly = np.ones_like(x, dtype=float)
    for j in range(x.shape[0]):
        for i in range(x.shape[0]):
            if (x[i] == x[j]): continue
            poly[j] *= (inp - x[i])/(x[j] - x[i])
    return np.einsum("j...,j->...", y, poly)

def propagator_matrix(c_in: np.ndarray, arg: np.ndarray):
    return expm(arg) @ c_in

def integrate_quantum(traj: Trajectory):
    """
    Calculates time-derivative couplings by various means and integrates quantum (electronic) EOMs

    JCC + JK 2024
    """

    #TODO create fix_phase function

    traj.est.coeff_mns[-1,0,:] = traj.est.coeff_mns[-2,0,:]

    for i in range(traj.par.n_states):
        traj.pes.overlap_mnss[-1,0,i,:] *= traj.pes.phase_s[i]


    # phase the overlaps here for now, but should create phasing module probably...
    phase_vec = np.ones(traj.par.n_states)
    for i in range(traj.par.n_states):
        if traj.pes.overlap_mnss[-1,0,i,i] < 0:
            phase_vec[i] *= -1
            traj.pes.overlap_mnss[-1,0,:,i] *= -1


    traj.pes.phase_s = phase_vec

    ddts = np.zeros((traj.par.n_qsteps,traj.par.n_states, traj.par.n_states))

    option = traj.est.tdc_updater
    if option == 'hst':
        # Classic Hammes-Schiffer-Tully mid point approximation (paper in 1994)
        ddts[:] = 1/(2*traj.ctrl.dt) * (traj.pes.overlap_mnss[-1,0,:,:] - traj.pes.overlap_mnss[-1,0,:,:].T)

    elif option == 'nacme':
        # ddt = nacme . velocity (i.e. original Tully 1990 paper model)
        for traj.ctrl.qstep in range(traj.par.n_qsteps):
            frac = (traj.ctrl.qstep+0.5)/traj.par.n_qsteps
            ddts[traj.ctrl.qstep] = frac*traj.pes.nac_ddt_mnss[-1,0] + (1-frac)*traj.pes.nac_ddt_mnss[-2,0]

    elif option == 'npi':
        # Meek and Levine's norm preserving interpolation, but integrated across the time-step 
        for traj.ctrl.qstep in range(traj.par.n_qsteps):
            frac = (traj.ctrl.qstep+0.5)/traj.par.n_qsteps
            U = np.eye(traj.par.n_states)*np.cos(np.arccos(traj.pes.overlap_mnss[-1,0,:,:])*frac)
            U += -1*(np.eye(traj.par.n_states)-1)*np.sin(np.arcsin(traj.pes.overlap_mnss[-1,0,:,:])*frac)
            dU = np.eye(traj.par.n_states)*(-np.sin(np.arccos(traj.pes.overlap_mnss[-1,0,:,:])*frac)*np.arccos(traj.pes.overlap_mnss[-1,0,:,:])/traj.ctrl.dt)
            dU += -1*(np.eye(traj.par.n_states)-1)*(np.cos(np.arcsin(traj.pes.overlap_mnss[-1,0,:,:])*frac)*np.arcsin(traj.pes.overlap_mnss[-1,0,:,:])/traj.ctrl.dt)

            ddts[traj.ctrl.qstep] = (U.T @ dU) * -1*(np.eye(traj.par.n_states)-1) # to get rid of non-zero diagonal elements

    elif option == 'npi_sharc':
        # NPI sharc mid-point averaged 
        Utot = np.zeros_like(traj.pes.overlap_mnss[-1,0,:,:])
        
        for i in range(traj.par.n_qsteps):
            U   =     np.eye(traj.par.n_states)    *   np.cos(np.arccos(traj.pes.overlap_mnss[-1,0,:,:])*i/traj.par.n_qsteps)
            U  += -1*(np.eye(traj.par.n_states)-1) *   np.sin(np.arcsin(traj.pes.overlap_mnss[-1,0,:,:])*i/traj.par.n_qsteps)
            dU  =     np.eye(traj.par.n_states)    * (-np.sin(np.arccos(traj.pes.overlap_mnss[-1,0,:,:])*i/traj.par.n_qsteps)*np.arccos(traj.pes.overlap_mnss[-1,0,:,:])/traj.ctrl.dt)
            dU += -1*(np.eye(traj.par.n_states)-1) * ( np.cos(np.arcsin(traj.pes.overlap_mnss[-1,0,:,:])*i/traj.par.n_qsteps)*np.arcsin(traj.pes.overlap_mnss[-1,0,:,:])/traj.ctrl.dt)
            Utot += np.matmul(U.T, dU)

        Utot /= traj.par.n_qsteps
        ddts[:] = Utot*(np.eye(traj.par.n_states)-1)*-1


    elif option ==  'npi_meek':
        # NPI Meek and Levine mid-point averaged
        def sinc(x):
            if np.abs(x) < 1e-9:
                return 1
            else:
                return np.sin(x)/x

        if traj.ctrl.curr_step > 2:
            w = traj.pes.overlap_mnss[-1,0,:,:]
            for k in range(traj.par.n_states):
                for j in range(traj.par.n_states):
                    if k == j:
                        continue
                    A = -sinc(np.arccos(w[j,j])-np.arcsin(w[j,k]))
                    B =  sinc(np.arccos(w[j,j])+np.arcsin(w[j,k]))
                    C =  sinc(np.arccos(w[k,k])-np.arcsin(w[k,j]))
                    D =  sinc(np.arccos(w[k,k])+np.arcsin(w[k,j]))
                    E = 0.
                    if traj.par.n_states != 2:
                        sqarg = 1-w[j,j]**2 - w[k,j]**2
                        if sqarg > 1e-6:
                            wlj = np.sqrt(sqarg)
                            wlk = -(w[j,k]*w[j,j] + w[k,k]*w[k,j])/wlj
                            if np.abs(wlk - wlj) > 1e-6:
                                E = wlj**2
                            else:
                                E = 2*np.arcsin(wlj)*(wlj*wlk*np.arcsin(wlj)+(np.sqrt((1-wlj**2)*(1-wlk**2))-1)*np.arcsin(wlk))/(np.arcsin(wlj)**2-np.arcsin(wlk)**2)

                    traj.pes.nac_ddt_mnss[-1,0,k,j] = 1/(2*traj.ctrl.dt) * (np.arccos(w[j,j])*(A+B)+np.arcsin(w[k,j])*(C+D) +E)
            ddts[:] = traj.pes.nac_ddt_mnss[-1,0,:,:]


    elif option == 'hst_sharc':
        # SHARC HST end-point finite difference, linearly interpolated across the region
        # Maybe don't trust this code too much...
        ddt_ini = 1/(4*traj.ctrl.dt) * (3*(traj.pes.overlap_mnss[-2,0,:,:] - traj.pes.overlap_mnss[-2,0,:,:].T)-(traj.pes.overlap_mnss[-3,0,:,:]-traj.pes.overlap_mnss[-3,0,:,:].T))
        ddt_fin = 1/(4*traj.ctrl.dt) * (3*(traj.pes.overlap_mnss[-1,0,:,:] - traj.pes.overlap_mnss[-1,0,:,:].T)-(traj.pes.overlap_mnss[-2,0,:,:]-traj.pes.overlap_mnss[-2,0,:,:].T))
        for traj.ctrl.qstep in range(traj.par.n_qsteps):
            frac = (traj.ctrl.qstep+0.5)/traj.par.n_qsteps
            ddts[traj.ctrl.qstep] = frac*ddt_ini+ (1-frac)*ddt_fin


    elif option == 'local_diabatisation' or option == 'ld':
        # local diabatisation
        # this is a bit of a hacky way to do it, not advised for beginners...
        # As the rest of the code all has the same structure (ie generate ddt and use that in the wavefunction propagation)
        # Here we separate the local diabatisation to its own function with a return
        if traj.ctrl.curr_step < 2:
            return
        R = np.eye(traj.par.n_states)
        H_tr = traj.pes.overlap_mnss[-1,0] @ traj.pes.ham_diag_mnss[-1,0] @ traj.pes.overlap_mnss[-1,0].T
        for traj.ctrl.qstep in range(traj.par.n_qsteps):
            frac = (traj.ctrl.qstep+0.5)/traj.par.n_qsteps
            H = traj.pes.ham_diag_mnss[-2,0] + frac*(H_tr-traj.pes.ham_diag_mnss[-2,0])

            R = expm(-1j*H*traj.ctrl.dt/traj.par.n_qsteps) @ R

        R = traj.pes.overlap_mnss[-1,0].T @ R

        traj.est.coeff_mns[-1,0] = R @ traj.est.coeff_mns[-1,0]
        if traj.par.type == "sh" and traj.hop.target == traj.hop.active: 
            get_hopping_prob_LD(traj, R)
            check_hop(traj)

        return
    else:
        print('The TDC updater requested is not implemented')
        raise ValueError

    # transform coeff into correct representation
    traj.est.coeff_mns[-1,0] = traj.pes.ham_transform_mnss[-1,0] @ traj.est.coeff_mns[-1,0]

    # propagation

    for traj.ctrl.qstep in range(traj.par.n_qsteps):
        # create variables for argument
        frac = (traj.ctrl.qstep+0.5)/traj.par.n_qsteps
        energy_ss = frac*traj.pes.ham_diag_mnss[-1,0] + (1-frac)*traj.pes.ham_diag_mnss[-2,0]

        # we use precomputed ddts
        arg = -(1.j*energy_ss + ddts[traj.ctrl.qstep])*traj.ctrl.dt/traj.par.n_qsteps
        
        #propagate using matrix exponential propagator

        traj.est.coeff_mns[-1,0] = traj.est.propagator(traj.est.coeff_mns[-1,0], arg)
        traj.pes.nac_ddt_mnss[-1,0] = ddts[traj.ctrl.qstep]

        # calculate hopping probability and hop if doing surface hopping
        if traj.par.type == "sh" and traj.hop.target == traj.hop.active:
            #continue
            get_hopping_prob_ddr(traj)
            check_hop(traj)

    # transform coeff back
    traj.est.coeff_mns[-1,0] = traj.pes.ham_transform_mnss[-1,0].conj().T @ traj.est.coeff_mns[-1,0]

def get_dt(traj: Trajectory):
    def get_inp(x,coeff):
        inp = 0
        for i in range(traj.par.n_states):
            for j in range(i):
                inp += np.sum(x[i,j]**2)*(np.abs(coeff[i])**2+np.abs(coeff[j])**2)
        inp = np.sqrt(inp)
        return inp
    
    traj.ctrl.h[-1] = traj.ctrl.h[-2]
    t = np.cumsum(traj.ctrl.h)

    f0 = traj.ctrl.dt_func(traj, get_inp(traj.pes.nac_ddt_mnss[-1,0], traj.est.coeff_mns[-1,0]))
    #  d = interpolate(t[:-1], traj.pes.nac_ddt_mnss[1:,0], t[-1])
    f1 = f0
    #  f1 = traj.ctrl.dt_func(traj, get_inp(d))
    traj.ctrl.h[-1] = 0.5*(f0 + f1)
    traj.ctrl.dt = traj.ctrl.h[-1]

"""def get_dt(traj: Trajectory):
    def get_inp(x):
        inp = 0
        for i in range(traj.par.n_states):
            for j in range(i):
                inp += np.sum(x[i,j]**2)
        inp = np.sqrt(inp)
        return inp
    print(traj.ctrl.h)
    print("TDC size: ", get_inp(traj.pes.nac_ddt_mnss[-1,0]))
    traj.ctrl.h[-1] = traj.ctrl.h[-2]
    t = np.cumsum(traj.ctrl.h)

    f0 = traj.ctrl.dt_func(traj, get_inp(traj.pes.nac_ddt_mnss[-1,0]))
    err = 1
    i = 0

    while err > 1e-8:
        i += 1
        t = np.cumsum(traj.ctrl.h)
        d = interpolate(t[:-1], traj.pes.nac_ddt_mnss[1:,0], t[-1])
        f1 = traj.ctrl.dt_func(traj, get_inp(d))
        temp = 0.5*(f0 + f1)
        print(f0, f1, temp)
        err = np.abs(temp - traj.ctrl.h[-1])
        print(get_inp(d), err)
        traj.ctrl.h[-1] = temp
        if i > 100: exit()
    traj.ctrl.dt = traj.ctrl.h[-1]
    print("Num iter: ", i)"""
