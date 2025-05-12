import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from classes.constants import units, convert
from electronic.models import Model

def pot(x):

    # HARMONIC

    a = 0.005
    b = 2
    c = 2

    ham = np.zeros((2,2))
    ham[0,0] = a * (x - b)**2
    ham[1,1] = a * (x + b)**2
    ham[0,1] = a*c
    ham[1,0] = a*c

    # SUB 1

    # a = 0.01
    # b = 0.6
    # c = 0.003
    # d = 1

    # ham = np.zeros((2,2))
    # ham[0,0] = a * np.tanh(b * x)
    # ham[1,1] = -ham[0,0]
    # ham[0,1] = c * np.exp(-d * x**2)
    # ham[1,0] = ham[0,1]

    # TULLY 1

    # a = 0.01
    # b = 1.6
    # c = 0.005
    # d = 1.0

    # ham = np.zeros((2,2))
    # if x > 0:
    #     ham[0,0] = a*(1 - np.exp(-b*x))
    # else:
    #     ham[0,0] = -a*(1 - np.exp(b*x))
    # ham[1,1] = -ham[0,0]
    # ham[0,1] = c*np.exp(-d*x**2)
    # ham[1,0] = ham[0,1]

    # SUB S

    # a = 0.015
    # b = 1.0
    # c = 0.005
    # d = 0.5

    # ham = np.zeros((3,3))
    # ham[0,0] = a*(np.tanh(b*(x - 7)) - np.tanh(b*(x + 7))) + a
    # ham[1,1] = -ham[0,0]
    # ham[2,2] = 3*a*np.tanh(b/2*x)
    # ham[0,1] = c*(np.exp(-d*(x - 7)**2) + np.exp(-d*(x + 7)**2))
    # ham[0,2] = c*np.exp(-d*x**2)
    # ham[1,2] = c*np.exp(-d*x**2)
    # ham[1,0] = ham[0,1]
    # ham[2,0] = ham[0,2]
    # ham[2,1] = ham[1,2]

    # SUB X

    # a = 0.03
    # b = 1.6
    # c = 0.005
    # d = 0.1

    # ham = np.zeros((3,3))
    # ham[0,0] = a*(np.tanh(b*x) + np.tanh(b*(x + 7)))
    # ham[1,1] = -a*(np.tanh(b*x) + np.tanh(b*(x - 7)))
    # ham[2,2] = -a*(np.tanh(b*(x + 7)) - np.tanh(b*(x - 7)))
    # ham[0,1] = c*np.exp(-d*x**2)
    # ham[0,2] = c*np.exp(-d*(x + 7)**2)
    # ham[1,2] = c*np.exp(-d*(x - 7)**2)
    # ham[1,0] = ham[0,1]
    # ham[2,0] = ham[0,2]
    # ham[2,1] = ham[1,2]

    return ham

def main():
    plot = 1
    n_points = 20000
    n_states = 2
    initstate = 1
    t = 0
    dt = 5
    tmax = 20000
    xmin = -20
    xmax = 20
    dx = (xmax - xmin)/(n_points - 1)
    x0 = -10
    m = convert(1, "amu", "au")
    E0 = 0.04
    p0 = np.sqrt(2*m*E0)
    print(p0)
    sigx = 0.5
    mid_idx = int(n_points/(xmax-xmin)*np.abs(xmin))

    Vd = np.zeros((n_points, n_states, n_states), dtype=np.complex128)
    x = np.linspace(xmin, xmax, n_points, endpoint=True)
    for n in range(n_points):
        Vd[n,:,:] = pot(x[n])

    xcutoff = [-15,15]

    l = x[-1] - x[0]
    grid = np.linspace(0, n_points-1, n_points, endpoint=True) - n_points/2
    kv = 2*np.pi/l * grid
    Va = np.zeros_like(Vd)
    avs = np.zeros_like(Vd)
    Vdexp = np.zeros_like(Vd)

    Vnip = -1j*np.heaviside(x-xcutoff[1], 0)*(x-xcutoff[1])**4 - 1j*np.heaviside(xcutoff[0]-x, 0)*(xcutoff[0]-x)**4

    for n in range(n_points):
        eval, evec = np.linalg.eig(Vd[n,:,:])
        idx = np.argsort(eval)
        eval = eval[idx]
        evec = evec[:,idx]
        for i in range(n_states):
            if n > 0:
                if np.sum(avs[n-1,:,i] * evec[:,i]) < 0:
                    evec[:,i] *= -1
            Va[n,i,i] = eval[i]
            avs[n,:,i] = evec[:,i]
        Vd[n,:,:] += np.eye(n_states)*Vnip[n]


    kve = np.exp(-1j*dt/2/m*kv**2)

    for n in range(n_points):
        Vdexp[n,:,:] = expm(-1j*dt*Vd[n,:,:])

    psi0 = np.exp(-(x-x0)**2/2/sigx**2 + 1j*(x-x0)*p0)
    norm = np.trapezoid(np.abs(psi0[:])**2, dx=dx)
    psi0 /= np.sqrt(norm)
    psid = np.zeros((n_points, n_states), dtype=np.complex128)
    psia = np.zeros_like(psid)
    # psid[:,initstate] = psi0
    psia[:,initstate] = psi0
    for n in range(n_points):
        psid[n,:] = np.linalg.inv(avs[n,:,:]) @ psia[n,:]

    psidft = np.zeros_like(psid)

    if plot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.set_xlim([xmin,xmax])
        axt = ax.twinx()
        axt.set_ylim([-2,2])

        for n in range(n_points):
            psia[n,:] = avs[n,:,:] @ psid[n,:]

        pes = []
        lns = []
        for state in range(n_states):
            pes.append(ax.plot(x, Va[:,state,state], 'k--')[0])
            lns.append(axt.plot(x, np.abs(psia[:,state])**2 - 1 + state)[0])
            # pes.append(ax.plot(x, Vd[:,state,state], '--')[0])
            # lns.append(axt.plot(x, np.abs(psid[:,state])**2 - 1 + state)[0])
        ax.plot(x, np.ones_like(x) * (1/2*p0**2/m + Va[0, initstate, initstate]), "r--")

    # breakpoint()
    out = open("qua.dat", "w")
    while t <= tmax/dt:
        print(t*dt)
        out.write(f"{convert(t * dt, "au", "fs")}")
        # for n in range(n_points):
        #     psia[n,:] = avs[n,:,:] @ psid[n,:]
        np.einsum("nij,nj->ni", avs, psid, out=psia)
        tot = 0
        for state in range(n_states):
            temp = np.trapezoid(np.abs(psia[:,state])**2, dx=dx)
            out.write(f" {temp}")
            tot += temp
        for state in range(n_states):
            out.write(f" {np.trapezoid(np.abs(psia[:n_points//2, state])**2, dx=dx)}")
            out.write(f" {np.trapezoid(np.abs(psia[n_points//2:, state])**2, dx=dx)}")
        out.write(f" {tot}\n")

        np.einsum("nij,nj->ni", Vdexp, psid, out=psid)

        np.fft.fft(psid, n_points, axis=0, out=psidft)
        psidft = np.fft.fftshift(psidft, axes=0)
        np.einsum("n,ni->ni", kve, psidft, out=psidft)
        np.fft.ifft(np.fft.fftshift(psidft, axes=0), n_points, axis=0, out=psid)

        if t % 10 == 0:
            # for n in range(n_points):
            #     psia[n,:] = avs[n,:,:] @ psid[n,:]

            if plot:
                for state in range(n_states):
                    lns[state].set_ydata(np.abs(psia[:,state])**2 - 1 + state)
                    # lns[state].set_ydata(np.abs(psid[:,state])**2 - 1 + state)
                fig.canvas.draw()
                fig.canvas.flush_events()
        t += 1
    out.close()

if __name__ == "__main__":
    main()