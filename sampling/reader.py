import numpy as np
import sys
import json
from classes.constants import convert

def osc_str(en, dm):
    e = np.diagonal(en)
    ediff = e[None,:] - e[:,None]

    return ediff, 2/3 * np.einsum('ij,ijk->ij', ediff, np.square(np.abs(dm)))


def main():
    with open(sys.argv[1], 'r') as file:
        config = json.load(file)

    samp = config["sampling"]

    # read one to get information
    try:
        en = np.load(f'{sys.argv[2]}/energy.npy')
        dm = np.load(f'{sys.argv[2]}/dipole.npy')
    except FileNotFoundError:
        print(f'File not found, continuing without IC {sys.argv[2]}')

    ens = np.zeros((len(sys.argv[2:]), *en.shape))
    oss = np.zeros_like(ens)

    for ii, i in enumerate(sys.argv[2:]):
        try:
            en = np.load(f'{i}/energy.npy')
            dm = np.load(f'{i}/dipole.npy')
        except FileNotFoundError:
            print(f'File not found, continuing without IC {i}')

        ens[ii], oss[ii] = osc_str(en, dm)

    emin = convert(samp["emin"], "au")
    emax = convert(samp["emax"], "au")

    st_ini = samp.get("from", 0)
    if isinstance(st_ini, int):
        st_ini = [st_ini]
    states_to_excite_from = np.array(st_ini)
    st_fin = samp.get("to", 1)
    if isinstance(st_fin, int):
        st_fin = [st_fin]
    states_to_excite_to = np.array(st_fin)
    allowed_transitions = (ens >= emin) * (ens <= emax)

    boss = oss * allowed_transitions

    max_osc = oss.max()

    s_oss = boss/max_osc
    # print(s_oss)

    fg = open('selected.xyz', 'w')
    fd = open('selected.dat', 'w')
    ft = open('total.dat', 'w')

    header = "#Icond   Directory             Initial   Final     Energy / eV       Osc. Str.    Selected?\n"
    fd.write(header)
    ft.write(header)

    for ini in states_to_excite_from:
        for fin in states_to_excite_to:
            if ini >= fin:
                    continue
            fg_s = open(f'selected_{ini}_{fin}.xyz', 'w')
            fd_s = open(f'selected_{ini}_{fin}.dat', 'w')
            fd_s.write(header)

            for ii, i in enumerate(sys.argv[2:]):
                chosen = s_oss[ii,ini,fin] > np.random.default_rng().random()
                s = f"{ii:4g}     {i:20s}  {ini:4g}     {fin:4g}   {ens[ii, ini, fin]:14.8f}    {oss[ii, ini, fin]:14.8f}   {chosen}\n"
                if chosen:
                    fd.write(s)
                    fd_s.write(s)
                    with open(f"{i}/geom.xyz") as fi:
                        ls = fi.readlines()
                        ls[1] = s
                        fg.write(''.join(ls))
                        fg_s.write(''.join(ls))

                ft.write(s)
            fg_s.close()
            fd_s.close()



    fg.close()
    fd.close()
    ft.close()


if __name__ == "__main__":
    main()