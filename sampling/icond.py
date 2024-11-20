
from classses.classes import ElectronicStructure, PotentialEnergySurface, Geometry, Params
from selection import select_est

from abinitio import run_molcas, run_molpro, set_est_mfe, set_est_sh, run_pyscf_wrapper, run_turbo, run_model
import json
import numpy as np
import sys


# NEEDS TO BE ADAPTED TO THE NEW FRAMEWORK
# should be sufficient to initialise a Molecule object / Icond can inherit from Molecule

class Icond:
    def __init__(self, config: dict):
        self.par = Params(config)
        self.par.n_steps = 1
        self.par.n_qsteps = 1

        self.est = ElectronicStructure(config, self.par)
        self.pes = PotentialEnergySurface(config, self.par)

        # remove non-important parts
        try:
            config['nuclear']['integrator'] = None
        except KeyError:
            config['nuclear'] = {}
            config['nuclear']['integrator'] = None

        self.geo = Geometry(config, self.par)


def main():

    filename = sys.argv[1]

    with open(filename, 'r') as f:
        config = json.load(f)

    icond = Icond(config)

    select_est(icond)
    print(icond.est.run)

    icond.est.run(icond)

    np.save('energy.npy', icond.pes.ham_diab_mnss[-1,0])
    np.save('dipole.npy', icond.pes.dip_mom_mnssd[-1,0])



if __name__ == '_main_':
    main()
