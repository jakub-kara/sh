import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import os, sys
from scipy.interpolate import CubicSpline, interp1d
from classes.constants import convert

def progress_bar(iteration, total, prefix="", suffix="", length=30, fill="="):
    percent = (f"{(100 * iteration / float(total)):.1f}")
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + " " * (length - filled_length)
    print(f"{prefix} [{bar}] {percent}% {suffix}", end="\r")

    if iteration == total:
        print()

def get_pos(pos, *idxs):
    out = []
    for idx in idxs:
        if idx == "o":
            out.append(np.zeros_like(pos[0]))
        else:
            out.append(pos[int(idx)])
    return out

def get_atom(atoms, idx):
    if idx == "o":
        return ""
    else:
        return atoms[int(idx)]

def get_file(dir, ext):
    files = [i for i in os.listdir(dir) if i.endswith(ext)]
    if len(files) == 0:
        print(f"No {ext} file found in {dir}.")
        exit()
    elif len(files) > 1:
        print(f"More than one {ext} file found in {dir}.")
        exit()
    return dir + files[0]

def get_dat_idx(do: bool, line1: str, line2: str, key: str):
    if not do:
        return None, do
    idx = line1.find(key)
    if idx == -1:
        print(f"{key} state data not found in the dat file. Skipping.")
        do = False
    return len(line2[:idx-1].split()), do

def validate_exclusive(parsed, num, *keys):
    if sum(parsed[key] is not None for key in keys) != num:
        raise KeyError(f"{num} arguments from " + ", ".join(f"--{key}" for key in keys) + " must be provided.")

def get_times(parsed, unit="fs"):
    n_steps = parsed.nsteps
    dt = parsed.dt
    tmax = parsed.tmax

    if n_steps is None:
        tmax = convert(tmax, unit)
        dt = convert(dt, unit)
        n_steps = int(tmax // dt)

    if dt is None:
        tmax = convert(tmax, unit)
        n_steps = int(n_steps)
        dt = tmax / n_steps

    if tmax is None:
        dt = convert(dt, unit)
        n_steps = int(n_steps)
        tmax = n_steps * dt

    return n_steps, dt, tmax

def get_alpha(x, min=0.2):
    return np.exp(-x + 1) * (1 - min) + min

parser = argparse.ArgumentParser()
parser.add_argument("--no-h5", action="store_true", default=False)
parser.add_argument("-i", "--trajs", action="store", nargs="*", required=True, default=[])
parser.add_argument("-o", "--output", action="store", default=None)

parser.add_argument("-t", "--tmax", action="store", default=None)
parser.add_argument("-n", "--nsteps", action="store", default=None)
parser.add_argument("-l", "--dt", action="store", default=None)

parser.add_argument("-a", "--angle", action="append", nargs=3, default=[])
parser.add_argument("-b", "--bond", action="append", nargs=2, default=[])
parser.add_argument("-c", "--active", action="store_true", default=False)
parser.add_argument("-d", "--dihedral", action="append", nargs=4, default=[])
parser.add_argument("-e", "--energy", action="store_true", default=False)
parser.add_argument("-p", "--pop", action="store_true", default=False)

parser.add_argument("-s", "--show", action="store_true", default=False)
parser.add_argument("-m", "--mean", action="store_true", default=False)

args = parser.parse_args()

validate_exclusive(vars(args), 2, "tmax", "nsteps", "dt")

trajs = args.trajs
if len(trajs) == 0:
    print("Please provide at least one trajectory")
    exit()

bonds = args.bond
angles = args.angle
dihs = args.dihedral

do_active = args.active
do_pops = args.pop
do_energy = args.energy

noh5 = args.no_h5

n_traj = len(trajs)

if noh5:
    n_steps, dt, tmax = get_times(args, "fs")
    with open(get_file(trajs[0] + "/data/", ".dat"), "r") as f:
        line1 = f.readline()
        n_states = line1.count("Population")
        line2 = f.readline()

        act_idx, do_active = get_dat_idx(do_active, line1, line2, "Active")
        pop_idx, do_pops = get_dat_idx(do_pops, line1, line2, "Population")
        en_idx, do_energy = get_dat_idx(do_energy, line1, line2, "Total En")

    atoms = []
    with open(get_file(trajs[0] + "/data/", ".xyz"), "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                n_atoms = int(line.strip())
                continue
            if i == 1:
                continue
            if i >= n_atoms + 2:
                break
            at = line.split()[0]
            atoms.append(at)
            n_dim = int((len(line.split()) - 1) // 2)
    atoms = np.array(atoms, dtype="<U2")

else:
    n_steps, dt, tmax = get_times(args, "au")
    f = h5py.File(get_file(trajs[0] + "/data/", ".h5"), "r")
    n_states = f["info/nst"][()]
    atoms = f["info/ats"][:].astype("<U2")
    n_atoms = len(atoms)
    n_dim = 3

times = np.arange(n_steps) * dt

nout = len(bonds) + len(angles) + len(dihs) + do_active*n_states + do_pops*n_states + do_energy
if nout == 0:
    print("Please request at least one value to analyse.")
    exit()

data = np.zeros((n_traj, n_steps, nout))
for itraj, traj in enumerate(trajs):
    progress_bar(itraj+1, n_traj, suffix=traj)

    if noh5:
        # temp = np.genfromtxt(get_file(traj + "/data/", ".xyz"), skip_header=2, invalid_raise=False, usecols=[i+1 for i in range(n_dim)])
        # pos = temp.reshape((-1,n_atoms,n_dim))

        with open(get_file(traj + "/data/", ".xyz"), "r") as f:
            lines = f.readlines()
            dur = len(lines) // (n_atoms + 2)
            val = np.zeros(dur)
            act = np.zeros(dur)
            pop = np.zeros((dur, n_states))
            pos = np.zeros((dur, n_atoms, n_dim))

            for i, line in enumerate(lines):
                if i % (n_atoms + 2) < 2:
                    at = 0
                    step = int(i // (n_atoms + 2))
                    continue
                temp = line.split()
                pos[step, at] = [float(x) for x in temp[1:n_dim+1]]
                at += 1

        temp = np.genfromtxt(get_file(traj + "/data/", ".dat"), skip_header=1)
        if temp.ndim == 1:
            continue
        val = temp[:,0]

        if do_active:
            act = temp[:,act_idx]

        if do_pops:
            pop = temp[:,pop_idx:pop_idx+n_states]

        if do_energy:
            en = temp[:,en_idx]
            en -= en[0]

    else:
        f = h5py.File(get_file(traj + "/data/", ".h5"), "r")

        dur = len(f.keys()) - 1
        val = np.zeros(dur)
        act = np.zeros(dur)
        en = np.zeros(dur)
        pop = np.zeros((dur, n_states))
        pos = np.zeros((dur, n_atoms, n_dim))

        for key in f.keys():
            if key == "info":
                continue
            step = int(key)
            val[step] = f[f"{key}/time"][()]
            pos[step] = f[f"{key}/pos"]

            if do_active:
                act[step] = f[f"{key}/act"][()]

            if do_pops:
                pop[step] = np.abs(f[f"{key}/coeff"])**2

            if do_energy:
                en[step] = f[f"{key}/toten"]

    if pos.ndim == 1:
        pos = pos.reshape((-1,1))
    pos = CubicSpline(val, pos, axis=0, extrapolate=False)

    if do_active:
        act = interp1d(val, np.array(act), kind="previous", bounds_error=False, fill_value=(act[0], act[-1]))

    if do_pops:
        pop = interp1d(val, np.array(pop), axis=0, bounds_error=False)

    if do_energy:
        en = interp1d(val, en, kind="previous", bounds_error=False)

    for step, time in enumerate(times):
        idx = 0

        if do_energy:
            data[itraj, step, idx] = en(time)
            idx += 1

        if do_active:
            temp = act(time)
            if np.isnan(temp):
                data[itraj, step, idx:idx+n_states] = np.nan
            else:
                data[itraj, step, idx + int(temp)] = 1
            idx += n_states

        if do_pops:
            data[itraj, step, idx:idx+n_states] = pop(time)
            idx += n_states

        for bond in bonds:
            r1, r2 = get_pos(pos(time), *bond)
            res = convert(np.linalg.norm(r1 - r2), "au", "aa")
            data[itraj, step, idx] = res
            idx += 1

        for ang in angles:
            r1, r2, r3 = get_pos(pos(time), *ang)
            v1 = r1 - r2
            v2 = r3 - r2
            res = np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))
            data[itraj, step, idx] = res
            idx += 1

        for dih in dihs:
            r1, r2, r3, r4 = get_pos(pos(time), *dih)
            v1 = r2 - r1
            v2 = r3 - r2
            v3 = r4 - r3
            # https://en.wikipedia.org/wiki/Dihedral_angle
            cr = np.cross(v2, v3)
            res = np.arctan2(np.linalg.norm(v2) * np.dot(v1, cr), np.dot(np.cross(v1, v2), cr))
            data[itraj, step, idx] = res
            idx += 1

alive = np.sum(1 - np.isnan(data[:,:,0]), axis=0)

if args.output:
    keys = []
    keys.extend(["toten"]*do_energy)
    keys.extend([f"popcl_{i}" for i in range(n_states)]*do_active)
    keys.extend([f"popqua_{i}" for i in range(n_states)]*do_pops)
    keys.extend([f"bond_{bond[0]}_{bond[1]}" for bond in bonds])
    keys.extend([f"angle_{angle[0]}_{angle[1]}_{angle[2]}" for angle in angles])
    keys.extend([f"dih_{angle[0]}_{angle[1]}_{angle[2]}_{angle[3]}" for angle in dihs])
    np.savez_compressed(args.output, time = times, trajs = alive, **{keys[i]: data[:,:,i] for i in range(nout)})

if args.show:
    idx = 0
    mean = np.nanmean(data, axis = 0)

    ax = plt.figure().add_subplot()
    ax.plot(times, alive)
    ax.set_title("Alive trajectories")

    if do_energy:
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(times, data[:, :, idx].T, c="r", alpha=get_alpha(n_traj))
        if args.mean:
            ax.plot(times, mean[:, idx], c="k")
        ax.set_title("Total Energy")

        ax = fig.add_subplot(212)
        den = np.zeros_like(data[:, :, idx])
        den[:, 1:] = data[:, 1:, idx] - data[:, :-1, idx]
        ax.plot(times, den.T, c="r", alpha=get_alpha(n_traj))
        if args.mean:
            ax.plot(times, np.nanmean(den, axis=0), c="k")
        ax.set_title("Total Energy Difference")
        idx += 1

    if do_active:
        ax = plt.figure().add_subplot()
        ax.plot(times, mean[:, idx:idx+n_states])
        ax.legend([f"State {i}" for i in range(n_states)])
        ax.set_title("Classical Populations")
        idx += n_states

    if do_pops:
        ax = plt.figure().add_subplot()
        ax.plot(times, mean[:, idx:idx+n_states])
        ax.legend([f"State {i}" for i in range(n_states)])
        ax.set_title("Quantum Populations")
        idx += n_states

    for (b1,b2) in bonds:
        ax = plt.figure().add_subplot()
        ax.plot(times, data[:, :, idx].T, c="r", alpha=get_alpha(n_traj))
        if args.mean:
            ax.plot(times, mean[:, idx], c="k")
        ax.set_title(f"{get_atom(atoms, b1)}{b1}-{get_atom(atoms, b2)}{b2} Bond Length")
        idx += 1

    for (a1,a2,a3) in angles:
        ax = plt.figure().add_subplot()
        ax.plot(times, data[:, :, idx].T, c="r", alpha=get_alpha(n_traj))
        if args.mean:
            ax.plot(times, mean[:, idx], c="k")
        ax.set_title(f"{get_atom(atoms, a1)}{a1}-{get_atom(atoms, a2)}{a2}-{get_atom(atoms, a3)}{a3} Bond Angle")
        idx += 1

    for (a1,a2,a3,a4) in dihs:
        ax = plt.figure().add_subplot()
        ax.plot(times, data[:, :, idx].T, c="r", alpha=get_alpha(n_traj))
        if args.mean:
            ax.plot(times, mean[:, idx], c="k")
        ax.set_title(f"{get_atom(atoms, a1)}{a1}-{get_atom(atoms, a2)}{a2}-{get_atom(atoms, a3)}{a3}-{get_atom(atoms, a4)}{a4} Dihedral Angle")
        idx += 1

    plt.show()