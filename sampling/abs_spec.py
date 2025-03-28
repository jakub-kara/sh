import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

ehtoev = 27.211386245988

### HELPER FUNCTIONS


def ev2nm(ev):
    return 4.135667696e-6 * 299792458 / ev


def gaussian(e, centre, height, fwhm):
    c = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return 1 / (c * np.sqrt(2 * np.pi)) * np.exp(-(e - centre)**2 /
                                                 (2 * c**2))


def lorentzian(e, centre, height, fwhm):
    g = 0.5 * fwhm
    return 1 / np.pi * g / ((e - centre)**2 + g**2)


def sech2(e, centre, height, fwhm):
    tau = fwhm / 1.7627
    a = 1 / np.cosh((e - centre) / tau)**2
    a /= np.linalg.norm(a)
    return a


### Main function


def plot_broadened_spectrum(fig, ax, excited, ini_states, fin_states, e_grid, save, fwhm, bf):

    ss_spectrum = np.zeros((len(ini_states), len(fin_states), len(e_grid)))
    tot_spectrum = np.zeros((len(e_grid)))



    for i, ini in enumerate(ini_states):
        for f, fin in enumerate(fin_states):
            boo = (excited[:, 0] == ini) * (excited[:, 1] == fin)
            ex = excited[boo, :]
            for q in range(ex.shape[0]):
                ss_spectrum[i, f, :] += bf(e_grid, ex[q, 2], ex[q, 3],
                                          fwhm)

    ss_spectrum /= excited.shape[0]

    ss_spectrum *= 2 * np.pi**2 * 0.0072973525693 * 0.52917721090**2

    tot_spectrum = np.sum(ss_spectrum, axis=(0, 1))

    ax.set_ylim([0, np.max(tot_spectrum) * 1.05])

    bottom = np.zeros_like(tot_spectrum)

    for i, ini in enumerate(ini_states):
        for f, fin in enumerate(fin_states):
            ax.fill_between(e_grid,
                            bottom,
                            bottom + ss_spectrum[i, f, :],
                            alpha=0.5,
                            label=f"S$_{{{ini}}}\\rightarrow$S$_{{{fin}}}$")
            bottom += ss_spectrum[i, f, :]

    ax.plot(e_grid, tot_spectrum, label='Total', color='#000000')

    if save:
        save_spectra(ss_spectrum, tot_spectrum, ini_states, fin_states, e_grid)

def plot_stick_spectrum(fig, ax, excited, ini_states, fin_states, e_grid, save):
    ax.set_ylabel('Osc. Str.')

    for i, ini in enumerate(ini_states):
        for f, fin in enumerate(fin_states):
            boo = (excited[:, 0] == ini) * (excited[:, 1] == fin)
            ex = excited[boo, :]
            ax.stem(ex[:,2], ex[:,3], markerfmt='')

    if save:
        print("Can't save stick spectra, continuing")

def save_spectra(ss_spectrum, tot_spectrum, ini_states, fin_states, e_grid):
    for i, ini in enumerate(ini_states):
        for f, fin in enumerate(fin_states):
            np.savetxt(
                f"spectrum_{ini}_{fin}.dat",
                np.hstack((e_grid[:, None], ss_spectrum[i, f, :][:, None])),
                header=
                f"E / eV                {ini} -> {fin} Cross-section / AA^2")

    np.savetxt(f"spectrum.dat",
               np.hstack((e_grid[:, None], tot_spectrum[:, None])),
               header=f"E / eV              Total Cross-section / AA^2")




def main():

    desc = '''
    Reads in initial conditions file and creates a broadened spectrum.

    Created by the Kirrander Group, University of Oxford

    '''

    parser = argparse.ArgumentParser(prog='abs_spec.py', description=desc)
    #  parser.add_argument('-n')
    parser.add_argument("filename",
                        type=str,
                        help='File containing initial condition lists')
    parser.add_argument('-s',
                        '--stick',
                        help='Plot stick spectrum instead',
                        action='store_true')
    parser.add_argument('-S',
                        '--save',
                        help='Save spectra into files - not for stick spectra',
                        action='store_true')
    parser.add_argument('-emin',
                        help='Minimum energy',
                        default=3,
                        type=float)
    parser.add_argument('-emax',
                        help='Maximum energy',
                        default=11,
                        type=float)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-l',
                        help='Use lorentzian',
                        action='store_true')
    group.add_argument('-g',
                       default=True,
                        help='Use gaussian',
                        action='store_true')
    parser.add_argument('-f',
                        '--fwhm',
                        help='FWHM in eV',
                        default=0.1,
                        type=int)
    parser.add_argument('-n',
                        help='Number of points',
                        default=1000,
                        type=int)



    args = parser.parse_args()

    filename = args.filename

    filename = sys.argv[1]

    excited = np.genfromtxt(filename, usecols=(2, 3, 4, 5))

    excited[:,2] *= ehtoev

    ini_states = set(excited[:, 0].astype(int))
    fin_states = set(excited[:, 1].astype(int))

    fig, ax = plt.subplots()
    ax.set_xlabel('E / eV')
    ax.set_ylabel('$\sigma$ / $\AA^2$')

    save = args.save
    
    fwhm = args.fwhm
    if args.l:
        bf = lorentzian
    else:
        bf = gaussian

    emin = args.emin
    emax = args.emax
    e_grid = np.linspace(emin, emax, num=args.n)

    if args.stick:
        plot_stick_spectrum(fig, ax, excited, ini_states, fin_states, e_grid, save)
    else:
        plot_broadened_spectrum(fig, ax, excited, ini_states, fin_states, e_grid, save, fwhm, bf)

    ax.set_xlim([e_grid[0], e_grid[-1]])

    ax.legend()
    ax.tick_params(top=False)
    secax = ax.secondary_xaxis('top', functions=(ev2nm, ev2nm))
    secax.set_xlabel('$\lambda$ / nm')
    secax.set_xticks([400, 300, 200, 100], minor=False)
    secax.set_xticks([450, 350, 250, 150, 50], minor=True)

    plt.show()

if __name__ == '_main_':
    main()
