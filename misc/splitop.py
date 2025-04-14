import numpy as np
from scipy.linalg import expm
from classes.constants import units, convert
import matplotlib.pyplot as plt
import sys

def pot(x):
    a = 0.005
    b = 2
    c = 2

    ham = np.zeros((2,2))
    ham[0,0] = a * (x - b)**2
    ham[1,1] = a * (x + b)**2
    ham[0,1] = a*c
    ham[1,0] = a*c
    return ham

def main():
    plot = 0
    n_points = 10000
    n_states = 2
    t = 0
    dt = 20
    tmax = 20000
    xmin = -15
    xmax = 15
    dx = (xmax - xmin)/(n_points - 1)
    x0 = -2
    p0 = 0
    sigx = 1/2
    mid_idx = int(n_points/(xmax-xmin)*np.abs(xmin))
    m = 1/units["amu"]

    Vd = np.zeros((n_points, n_states, n_states), dtype=np.complex128)
    x = np.linspace(xmin, xmax, n_points, endpoint=True)
    for n in range(n_points):
        Vd[n,:,:] = pot(x[n])

    xcutoff = [-15, 15]

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
            Va[n,i,i] = eval[i]
            avs[n,:,i] = evec[:,i]
        Vd[n,:,:] += np.eye(n_states)*Vnip[n]


    kve = np.exp(-1j*dt/2/m*kv**2)

    for n in range(n_points):
        Vdexp[n,:,:] = expm(-1j*dt*Vd[n,:,:])

    psi0 = np.exp(-(x-x0)**2/2/sigx**2 + 1j*(x-x0)*p0)
    norm = np.trapz(np.abs(psi0[:])**2, dx=dx)
    psi0 /= np.sqrt(norm)
    psid = np.zeros((n_points, n_states), dtype=np.complex128)
    psia = np.zeros_like(psid)
    psia[:,1] = psi0
    for n in range(n_points):
        psid[n,:] = np.linalg.inv(avs[n,:,:]) @ psia[n,:]

    psidft = np.zeros_like(psid)

    if plot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([-1,1])
        axt = ax.twinx()

        for n in range(n_points):
            psia[n,:] = avs[n,:,:] @ psid[n,:]

        pes1, = axt.plot(x, Va[:,0,0], 'g--')
        pes2, = axt.plot(x, Va[:,1,1], 'k--')
        line1, = ax.plot(x, np.abs(psia[:,0])**2, 'r')
        line2, = ax.plot(x, np.abs(psia[:,1])**2, 'b')

    # breakpoint()
    out = open("qua.dat", "w")
    while t <= tmax/dt:
        print(t*dt)
        out.write(f"{convert(t * dt, "au", "fs")}")
        for n in range(n_points):
            psia[n,:] = avs[n,:,:] @ psid[n,:]
        out.write(f" {np.trapz(np.abs(psia[:,0])**2, dx=dx)}")
        out.write(f" {np.trapz(np.abs(psia[:,1])**2, dx=dx)}\n")

        for n in range(n_points):
            psid[n,:] = Vdexp[n,:,:] @ psid[n,:]

        psidft = np.fft.fft(psid, n_points, axis=0)
        psidft = np.fft.fftshift(psidft, axes=0)
        for i in range(n_states):
            psidft[:,i] = kve*psidft[:,i]
        psid = np.fft.ifft(np.fft.fftshift(psidft, axes=0), n_points, axis=0)

        # if t % 1 == 0:
        #     for n in range(n_points):
        #         psia[n,:] = avs[n,:,:] @ psid[n,:]

        #     if plot:
        #         line1.set_ydata(np.abs(psia[:,0])**2)
        #         line2.set_ydata(np.abs(psia[:,1])**2)
        #         fig.canvas.draw()
        #         fig.canvas.flush_events()
        t += 1
    out.close()

if __name__ == "__main__":
    main()