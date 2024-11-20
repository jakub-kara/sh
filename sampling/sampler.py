#!python
import argparse
import scipy.special as sps
import scipy.integrate as spi
import scipy.signal as spss
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

evtocm = 8065.54429  # (cm eV)^-1
ehtoev = 27.21138624598
k_b = 0.695034800  # in (cm K)^-1
amu = 1822.8884847700401
cm2bohr = 5.29177210903e-9
c_cm = 29979245800
atu = 2.4188843265857e-17
k_b = 3.166811563e-6 # boltzmann in hartree

mass = {'O': 15.9999, 'H': 1.007825, 'C' : 12.0, 'e' : 2/amu , 'X' : 1e31}

def get_temp(ehmodes, U_vib):

    print(f"Target Energy = {U_vib:7.4f} eV")
    U_vib *= evtocm
    modes = ehtoev * evtocm * ehmodes
    print(f"Target Energy = {U_vib:7.4f} cm^-1")
    
    k_b = 0.695034800  # in (cm K)^-1
    
    def diff(F, x):
        # simple finite difference gradient calculation
        h = 0.001
        return 1/h * (F(x+h) - F(x))

    def part_func(T):
        # calculates partition function as the product of the individual modes
        return np.prod(1/(1-np.exp(-modes/(k_b*T))))

    def lp(T):
        # simple function to calculate log of function
        return np.log(part_func(T))

    def t2qlnqdt(T):
        return np.sum((modes/k_b)/(np.exp(modes/(k_b*T)) - 1))

    def get_e(T):
        # total energy = k_b T^2 (dln(q)/d(T))_V
        return k_b * t2qlnqdt(T)  # T**2 * diff(lp, T)
    
    def get_t(T):
        # Function that returns difference between wanted energy and energy at current T
        U_test = get_e(T[0])
        print(
            f"Curr. Energy = {U_test:7.4f} err {U_test-U_vib:7.4f} cm^-1, Curr. Temp. = {T[0]:7.4f} K")
        return U_test-U_vib
    
    def get_e2(T):
        return np.sum([mode/(np.exp(mode/(k_b*T)) - 1) for mode in modes])
    
    def get_t2(T):
        U_test = get_e2(T[0])
        return U_test-U_vib
    
    print('Mode  Wavenumbers (cm^-1)  Vib. Temp. (K)')
    for i, mode in enumerate(modes):
        print(f"{i:3g}      {mode:>7.4f}     {mode/k_b:>7.4f}")
    
    T_classical= get_classical_temp(U_vib, modes)
    print(f"Init. guess of classical temperature {T_classical:7.4f} K")
    
    T= spo.fsolve(get_t, T_classical, xtol=1e-10)[0]
    #second solve
    #  T2= spo.fsolve(get_t2, T_classical, xtol=1e-10)[0]

    print(f"Final Energy = {get_e(T):7.4f} cm^-1, Final Temp. = {T:7.4f} K")
    #  print(f"Final Energy = {get_e2(T2):7.4f} cm^-1, Final Temp. = {T2:7.4f} K")
    
    print(f"Classical temperature = {T_classical:7.4f} K")
    print(f"Quantum temperature   = {T:7.4f} K")
    #  print(f"Quantum temperature 2 = {T2:7.4f} K")
    
    # generate excitation lists

    return T

def get_classical_temp(U, modes):
    # equipartition energy (2 quadratic degrees of freedom)
    k_b = 0.695034800  # in (cm K)^-1
    return U/(len(modes)*k_b)

def write_xyz(filename, atoms, geoms, velocs, print_v):
    s, a, d = geoms.shape
    with open(filename, 'w') as f:
        for i in range(s):
            f.write(f'{a}\n\n')
            for j in range(a):
                f.write(f"{atoms[j]}   {geoms[i,j,0]:22.14f}  {geoms[i,j,1]:22.14f}  {geoms[i,j,2]:22.14f}" + print_v*f"  {velocs[i,j,0]:22.14f}   {velocs[i,j,1]:22.14f}     {velocs[i,j,2]:22.14f}" + "\n")

def read_molden(filename):
    atoms, coord, mass = get_geom(filename)
    mode_freqs, modes = get_modes(filename, len(atoms))
    return atoms, coord, mass, mode_freqs, modes

def get_geom(filename):
    atoms = []
    coord = []
    with open(filename, 'r') as f:
        for line in f:
            if '[FR-COORD]' in line:
                while '[' not in (l := f.readline()):
                    atoms.append(l.split()[0])
                    coord.append([float(q) for q in l.split()[1:]])
                break

    return atoms, np.array(coord), np.array([mass[i] * amu for i in atoms])

def get_modes(filename, no_atoms):
    # read in wavenumbers of vibrations from molden file
    freqs = []
    with open(filename, 'r') as f:
        for line in f:
            if '[FREQ]' in line.upper():
                while True:
                    l = f.readline()
                    if '[' in l.upper():
                        break
                    freq = float(l)
                    freqs.append(float(l))
                break

    no_modes = len(freqs)

    modes = np.zeros((no_modes, no_atoms, 3))
    with open(filename, 'r') as f:
        for line in f:
            if '[FR-NORM-COORD]' in line:
                print(line)
                for i in range(no_modes):
                    f.readline()
                    for j in range(no_atoms):
                        modes[i,
                              j, :] = [float(q) for q in f.readline().split()]
                break
    #  print(np.array(freqs)/(ehtoev*evtocm))

    f_bool = np.array(freqs) > 10 #cm 
    return np.array(freqs)[f_bool]/(ehtoev*evtocm), np.array(modes)[f_bool,:,:]

def get_overlap(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def clean_modes(modes, mass):
    thr = 1e-4
    s = get_overlap(modes[0]*np.sqrt(mass[:,None]), modes[1]*np.sqrt(mass[:,None]))
    if (np.abs(s)) < thr:
        print('correct when multiplying by sqrt(mass)')
        return modes * np.sqrt(mass)[:,None], modes
    s = get_overlap(modes[0], modes[1])
    if (np.abs(s)) < thr:
        print('already correct')
        return modes, modes / np.sqrt(mass)[:, None]
    s = get_overlap(modes[0]/np.sqrt(mass[:,None]), modes[1]/np.sqrt(mass[:,None]))
    if (np.abs(s)) < thr:
        print('correct when dividing by sqrt(mass)')
        return modes / np.sqrt(mass)[:,None], modes / mass[:,None]

def get_mass(modes, mass):
    return 1/np.einsum('ijk,j->i',  modes**2, 1/ mass)

def g_heller_width(omega, beta):
    return 1/(np.sqrt(2*np.tanh(beta*omega/2)))

def g_heller(H, omega, beta):
    # J. Chem. Phys. 65, 1289–1298 (1976)
    # https://doi.org/10.1063/1.433238
    # Apparently also
    # M. Hillery, R. F. O’Connell, M. O. Scully and E. P. Wigner,
    # Phys. Rep., 1984, 106, 121–167.
    return 1/np.pi * np.tanh(beta*omega/2) * np.exp(-2 * np.tanh(beta*omega/2)*H)

def g_wigner(H, omega, n):
    # wikipedia lol
    return ((-1)**n)/np.pi * lag(4*H, n) * np.exp(-2*H)
    #  return ((-1)**n)/np.pi * lag(4*H/omega, n) * np.exp(-2*H/omega)

def lag(x, n):
    #returns nth order laguerre polynomial value at x.
    return sps.laguerre(n)(x)

def g_harmonic(x, n):
    return 1/(2**n * np.math.factorial(n)) * np.pi**(-1/2) *  np.exp(-x**2) * sps.hermite(n)(x)**2
def g_husimi(H, omega, n):
    return (2*H)**n/(np.pi*np.math.factorial(n)) * np.exp(-2*H)
    #  return (2*H/omega)**n/(np.pi*np.math.factorial(n)) * np.exp(-2*H/omega)

def gaussian(x, sigma):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2 / sigma**2)

def sample(sampling_type, n_samples, coord, temperature, ns, cart_modes, mode_freqs, analyse, ignore):
    n_modes = len(mode_freqs)
    geoms = np.zeros((n_samples, *coord.shape))
    velocs = np.zeros((n_samples, *coord.shape))
    geoms[:,:,:] = coord[None, :,:]

    displacements = np.zeros((n_modes,n_samples,2))

    try:
        beta = 1/(k_b*temperature)
    except ZeroDivisionError:
        beta = None

    ks_test_1 = np.NaN
    ks_test_2 = np.NaN
    if sampling_type == 1: # returns wigner transform for T>=0
        print(f"Sampling the Wigner distribution with T = {temperature} K")
        print("Mode   Energy / Eh   Ana. Std    Samp. Std   Kol.-Smi. Q  Kol.-Smi. P")
        for i, omega in enumerate(mode_freqs):
            if ignore[i] == 1:
                print(f'Ignoring mode {i}')
                continue
            # half in front of Q and P in H cancels out with 2 in front of H
            if temperature > 0.:
                sigma = g_heller_width(omega, beta)
            else:
                sigma =  1/np.sqrt(2)

            displacements[i,:,:] = np.random.default_rng().normal(scale=sigma, size=(n_samples,2))

            if analyse:
                x = np.linspace(-10,10,5000)
                dx = x[1] - x[0]
                CDF = np.cumsum(gaussian(x, sigma))[1:]*dx
                hist, _ = np.histogram(displacements[i,:,0],bins=x,density=True)
                CDF2 = np.cumsum(hist)*dx
                ks_test_1 = np.max(np.abs(CDF2-CDF))
                hist, _ = np.histogram(displacements[i,:,1],bins=x,density=True)
                CDF2 = np.cumsum(hist)*dx
                ks_test_2 = np.max(np.abs(CDF2-CDF))

            print(f"{i:4g}  {omega:10.7f}    {sigma:10.7f}   {np.std(displacements[i]):10.7f}  {ks_test_1:10.7f}  {ks_test_2:10.7f}")
    elif sampling_type == 2: # Husimi
        print(f"Sampling the Husimi distribution with v = {ns}")
        print("Mode   Energy / Eh   v   Kol.-Smi. r  Kol.-Smi. theta   PDF int.")
        r = np.linspace(0,20,5000)
        dr = r[1]-r[0]
        for i, omega in enumerate(mode_freqs):
            rs = []
            n = ns[i]
            PDF = 2*np.pi*r**(2*n+1) / (np.pi * np.math.factorial(n)) * np.exp(-r**2)
            CDF = np.cumsum(PDF)*dr
            r_rand = np.random.default_rng().random(size=n_samples)
            phi_rand = np.random.default_rng().random(size=n_samples)*2*np.pi
            for ip, p in enumerate(r_rand):
                r_val = r[np.abs((CDF-p)).argmin()]
                displacements[i,ip,:] = r_val * np.array([np.cos(phi_rand[ip]),np.sin(phi_rand[ip])])
                if analyse:
                    rs.append(r_val)
            if analyse:
                hist, _ = np.histogram(rs, bins=r, density=True)
                CDF2 = np.cumsum(hist)*dr
                ks_test_1 = np.max(np.abs(CDF2-CDF[1:]))
                phis = np.linspace(0,2*np.pi,4000)
                hist, _ = np.histogram(phi_rand, bins=phis, density=True)
                CDF2 = np.cumsum(hist)*(phis[1]-phis[0])
                ks_test_2 = np.max(np.abs(CDF2-phis[1:]/(2*np.pi)))

            print(f"{i:4g}  {omega:10.7f}  {n:4g}  {ks_test_1:10.7f}  {ks_test_2:10.7f}  {CDF[-1]:7.4f}")
            if CDF[-1] < 1-1e-5:
                print('CAREFUL - you might need to increase the width of x!')

    elif sampling_type == 3: #Harmonic oscillator in position space (no momentum)
        print(f"Sampling the Harmonic oscillator distribution with v = {ns}")
        print("Mode   Energy / Eh   v   Kol.-Smi. x   PDF int.")
        x = np.linspace(-20,20,5000)
        dx = x[1]-x[0]
        for i, omega in enumerate(mode_freqs):
            xs = []
            n = ns[i]
            if n < 0:
                print(f"ignoring mode {i}")
                continue
            PDF = g_harmonic(x, n)
            CDF = np.cumsum(PDF)*dx
            x_rand = np.random.default_rng().random(size=n_samples)
            for ip, p in enumerate(x_rand):
                x_val = x[np.abs((CDF-p)).argmin()]
                displacements[i,ip,0] = x_val
                if analyse:
                    xs.append(x_val)
            if analyse:
                hist, _ = np.histogram(xs, bins=x, density=True)
                CDF2 = np.cumsum(hist)*dx
                np.savetxt('t2', CDF2)
                np.savetxt('t1', CDF)
                ks_test_1 = np.max(np.abs(CDF2-CDF[1:]))
            print(f"{i:4g}  {omega:10.7f}  {n:4g}  {ks_test_1:10.7f}  {CDF[-1]:10.7f}")
            if CDF[-1] < 1-1e-5:
                print('CAREFUL - you might need to increase the width of x!')

    # This manipulation is a little bit subtle, so here goes.
    # The modes provided by a electronic structure package have (implicitly) a reduced mass, expressed in atomic units. One must divide by this to get a unitless thing.
    # The mode_freqs come in to correct for the displacements being in natural units sqrt(hbar/m*omega), and the m in that equation gets swallowed into the cart_modes

    geoms += np.einsum('li,ljk,l->ijk',  displacements[:,:,0], cart_modes/np.sqrt(amu), 1/np.sqrt(mode_freqs))
    velocs = np.einsum('li,ljk,l->ijk',  displacements[:,:,1], cart_modes/np.sqrt(amu),   np.sqrt(mode_freqs))

    return geoms, velocs

def plot_functions(q,p,sampling_type, temperature, omega, n, d3, cmap):
    print(n)
    Q, P = np.meshgrid(q, p)

    H = 0.5 * (P**2+Q**2)

    try:
        beta = 1/(k_b*temperature)
    except ZeroDivisionError:
        beta = 1e-10
    omega = 1

    import matplotlib.pyplot as plt
    fig = plt.figure()
    if sampling_type == 1:
        lab = f'$W(Q, P), \\beta={beta:7.4f}$ Eh'
        d = g_heller(H, omega, beta)
        if d3:
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(Q, P, d, cmap=cmap)
            ax.set_zlabel(lab)
        else:
            ax = fig.add_subplot()
            tocb = ax.pcolormesh(Q, P, d, cmap=cmap)
            #  tocb = ax.contourf(Q, P, d, cmap=cmap)
            cbar = plt.colorbar(tocb)
            cbar.set_label(lab)
            #  ax.set_title(f"Wigner function, T = {temperature} K")
        ax.set_ylabel('$P$')
    elif sampling_type == 2:
        lab = f'$\\mathcal{{Q}}_{{{n}}}(Q, P)$'
        d = g_husimi(H, omega, n)
        if d3:
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(Q, P, d, cmap=cmap)
            ax.set_zlabel(lab)
        else:
            ax = fig.add_subplot()
            tocb = ax.pcolormesh(Q, P, d, cmap=cmap)
            #  tocb = ax.contourf(Q, P, d, cmap=cmap)
            #  ax.set_title(f"Husimi Q function")
            cbar = plt.colorbar(tocb)
            cbar.set_label(lab)
        ax.set_ylabel('$P$')

    elif sampling_type == 3:
        d = g_harmonic(q, n)
        if d.sum()*(q[1]-q[0]) < 0.99:
            q *= 2
            d = g_harmonic(q, n)
        ax = fig.add_subplot()
        ax.plot(q, d)
        ax.set_ylabel(f'$|\\psi_{{{n}}}(Q)|^2$')

    ax.set_xlabel('$Q$')

    plt.show()


def main():

    desc = '''
    Script for sampling vibrational wavefunctions in phase space.
    Based on a normal mode formalism, read from molden files.

    Created by the Kirrander Group, University of Oxford

    '''

    parser = argparse.ArgumentParser(prog='sampler.py',description=desc)
    #  parser.add_argument('-n')
    parser.add_argument("filename",type=str,help='Molden file containing normal modes')
    parser.add_argument('-d', help='Distribution: 1 for Wigner, 2 for Husimi, 3 for Harmonic Oscillator (no momentum)',type=int,choices=[1,2,3],default=0)
    parser.add_argument('-n', help='Number of samples',type=int,default=1000)
    parser.add_argument('-A', help='Also print standard xyz in AAngstrom', action='store_true')
    parser.add_argument('-a', help='Perform statistical analysis of sampling (Kolmogorov-Smirnov test)', action='store_true')
    parser.add_argument('-P', help='Plot functions sampled from for given mode', default = 0, type=int)
    parser.add_argument('--cmap', help='Colourmap for plotting', default = 'Reds', type=str)
    parser.add_argument('--jet', action='store_true', help=argparse.SUPPRESS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-T', help='Temperature in Kelvin - Program will use as temperature for Wigner sampling. Mutually exclusive with -E and -v',type=float,default=0.)
    group.add_argument('-E', help='Internal energy in eV - Program will calculate temperature and use. Mutually exclusive with -T and -v', type=float, default=0.)
    group.add_argument('-v', help='Number of quanta in each vibrational mode. A single number will be all modes in that state. More than one number will assign to each mode. Mutually exclusive with -T and -E. Note, this has to be the last argument (after filename)', nargs='+', type=int, default=[0])
    parser.add_argument('-i', help='!MUST BE AT THE END! Ignores these modes, leaving them at their equilibrium geometry. List of 3N-6 modes, 1 ignores the mode, 0 does not.', nargs='+', type=int, default=[0])
    
    args = parser.parse_args()

    filename = args.filename
    atoms, coord, mass, mode_freqs, init_modes = read_molden(filename)

    sampling_type = args.d
    n_samples = args.n
    analyse = args.a

    temperature = 0.
    v_quanta = np.zeros_like(mode_freqs, dtype=int)
    if args.T != 0:
        temperature = args.T
    elif args.E != 0:
        internal_energy = args.E
        temperature = get_temp(mode_freqs, internal_energy)
    elif len(args.v) > 1:
        if sampling_type == 1:
            print('Incompatible arguments - -v must be used with -d {2,3}')
            raise ValueError
        v_quanta = args.v
    elif args.v[0] != 0:
        if sampling_type == 1:
            print('Incompatible arguments - -v must be used with -d {2,3}')
            raise ValueError
        v_quanta = args.v*np.ones_like(mode_freqs, dtype=int)
    else:
        temperature = 0.

    ignore = args.i
    if len(ignore) != len(mode_freqs):
        ignore = np.zeros_like(mode_freqs)

    if temperature != 0.:
        if sampling_type != 1:
            print('Incompatible arguments - -E and -T must be used with -d 1')
            raise ValueError

    #THIS IS SPECIAL FOR DIATOMICS
    try:
        mw_modes, cart_modes = clean_modes(init_modes, mass)
    except IndexError:
        #For diatomics, expect molcas style input currently
        print('DIATOMIC from MOLCAS/MOLPRO only')
        mw_modes = init_modes * np.sqrt(mass)[:,None]
        cart_modes = init_modes

    r_m = get_mass(mw_modes, mass)
    print('r_m', r_m)

    # Begin sampling
    geoms, velocs = sample(sampling_type, n_samples, coord, temperature, v_quanta, cart_modes, mode_freqs, analyse, ignore)
    
    if args.A:
        write_xyz('wigner_AA.xyz', atoms, geoms*cm2bohr*1e8, velocs, False)

    write_xyz('wigner_au.xyz', atoms, geoms,velocs, True)

    print(args.P)
    cmap = args.cmap
    if args.jet:
        cmap = 'jet'
    plot  = args.P 
    if plot != 0:
        d3 = plot < 0
        plot = np.abs(plot)
        q = np.linspace(-4,4,2001)
        p = np.linspace(-4,4,2001)
        plot_functions(q, p, sampling_type, temperature, mode_freqs[plot], v_quanta[plot],d3,cmap)

if __name__ == '_main_':
    main()
