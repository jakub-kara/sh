import numpy as np

from classes import Trajectory
from abinitio import run_molcas, run_molpro, set_est_mfe, set_est_sh, run_pyscf_wrapper, run_turbo, run_model
from model_est import harm, spin_boson, tully_1, tully_2, tully_3, tully_n, tully_s, sub_2, sub_s, sub_x, lvc_wrapper
from errors import *
from integrators import SV, OV, RKN4, RKN6, RKN8, SY4, SY6, SY8, SY8b, SY8c, AM4, AM6, AM8, VVSolver, OVSolver, RKNSolver, ARKN3Solver, SYSolver, update_force_mfe, update_force_sh, propagator_matrix

def select_est(traj: Trajectory):
    if traj.est.program == "model":
        if "harm" in traj.est.type:
            traj.est.run = harm
        elif "sb" in traj.est.type:
            traj.est.run = spin_boson
        elif "tully_1" in traj.est.type:
            traj.est.run = tully_1
        elif "tully_s" in traj.est.type:
            traj.est.run = tully_s
        elif "tully_n" in traj.est.type:
            traj.est.run = tully_n
        elif "tully_2" in traj.est.type:
            traj.est.run = tully_2
        elif "tully_3" in traj.est.type:
            traj.est.run = tully_3
        elif "sub_x" in traj.est.type:
            traj.est.run = sub_x
        elif "sub_s" in traj.est.type:
            traj.est.run = sub_s
        elif "sub_2" in traj.est.type:
            traj.est.run = sub_2
        elif "lvc" in traj.est.type:
            from lvc import LVC
            traj.est.run = lvc_wrapper
        else:
            raise EstTypeNotFoundError
    elif traj.est.program == "molpro":
        traj.est.run = run_molpro
    elif traj.est.program == "molcas":
        traj.est.run = run_molcas
    elif traj.est.program == "pyscf":
        traj.est.run = run_pyscf_wrapper
    elif traj.est.program == "ricc2":
        traj.est.run = run_turbo
    else:
        raise EstProgramNotFoundError
    
def select_solvers(traj: Trajectory):
    solvers = {
        "vv": (RKNSolver, RKN4, 0, VVSolver, SV),
        "ov": (RKNSolver, RKN4, 0, OVSolver, OV),
        "arkn3": (RKNSolver, RKN4, 2, ARKN3Solver, SY4),
        "rkn4": (RKNSolver, RKN4, 0, RKNSolver, RKN4),
        "rkn6": (RKNSolver, RKN6, 0, RKNSolver, RKN6),
        "rkn8": (RKNSolver, RKN8, 0, RKNSolver, RKN8),
        "sy4": (RKNSolver, RKN4, 4, SYSolver, SY4),
        "sy6": (RKNSolver, RKN6, 6, SYSolver, SY6),
        "sy8": (RKNSolver, RKN8, 8, SYSolver, SY8),
        "sy8b": (RKNSolver, RKN8, 8, SYSolver, SY8b),
        "sy8c": (RKNSolver, RKN8, 8, SYSolver, SY8c)
    }
    temp = solvers.get(traj.geo.scheme_name)
    if temp is None: raise SolverTypeNotFoundError
    traj.geo.init_solver, traj.geo.init_scheme, traj.ctrl.init_steps, traj.geo.loop_solver, traj.geo.loop_scheme = temp

def select_force_updater(traj: Trajectory):
    updaters = {
        "sh": update_force_sh,
        "mfe" : update_force_mfe
    }
    temp = updaters.get(traj.par.type)
    if temp is None: raise TrajectoryTypeNotFoundError
    traj.geo.force_updater = temp

def select_nac_setter(traj: Trajectory):
    nac_setters = {
        "sh": set_est_sh,
        "mfe": set_est_mfe
    }
    temp = nac_setters.get(traj.par.type)
    if temp is None: raise TrajectoryTypeNotFoundError
    traj.est.nacs_setter = temp

def select_coeff_propagator(traj: Trajectory):
    propagators = {
        "propmat" : propagator_matrix,
    }
    temp = propagators.get(traj.est.propagator_name)
    if temp is None: raise PropagatorTypeNotFoundError
    traj.est.propagator = temp

def const(traj: Trajectory, inp: float):
    return traj.ctrl.dt_max

def gauss(traj: Trajectory, inp: float):
    temp = -inp**2/2/traj.ctrl.dt_params[0]**2
    return (traj.ctrl.dt_max - traj.ctrl.dt_min) * np.exp(temp) + traj.ctrl.dt_min

def tanh(traj: Trajectory, inp: float):
    temp = inp/traj.ctrl.dt_params[0]
    return (traj.ctrl.dt_max - traj.ctrl.dt_min) * (1 - np.tanh(temp-3))/2 + traj.ctrl.dt_min

def logistic(traj: Trajectory, inp: float):
    temp = -inp**2/2/traj.ctrl.dt_params[0]**2
    return (traj.ctrl.dt_max - traj.ctrl.dt_min) * 2 * np.exp(temp)/(1 + np.exp(temp)) + traj.ctrl.dt_min
    
def select_stepfunc(traj: Trajectory):
    stepfuncs = {
        "const": const,
        "gauss": gauss,
        "logistic": logistic,
        "tanh": tanh
    }
    temp = stepfuncs.get(traj.ctrl.dt_name)
    if temp is None: raise StepFunctionNotFoundError
    traj.ctrl.dt_func = temp

