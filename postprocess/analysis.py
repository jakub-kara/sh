import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
from classes.constants import convert

def get_pos(pos, *idxs):
    out = []
    for idx in idxs:
        if idx == "o":
            out.append(np.zeros_like(pos[0]))
        else:
            out.append(pos[int(idx)])
    return out

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--active", action="store_true", default=False)
parser.add_argument("-p", "--pop", action="store_true", default=False)
parser.add_argument("-b", "--bond", action="append", nargs=2, default=[])
parser.add_argument("-a", "--angle", action="append", nargs=3, default=[])
parser.add_argument("-d", "--dihedral", action="append", nargs=4, default=[])
parser.add_argument("-i", "--trajs", action="store", nargs="*", required=True, default=[])
parser.add_argument("-o", "--output", action="store", default=None)
parser.add_argument("-s", "--show", action="store_true", default=False)
parser.add_argument("-m", "--mean", action="store_true", default=False)
parser.add_argument("-v", "--verbose", action="store_true", default=False)
args = parser.parse_args()

trajs = args.trajs
if len(trajs) == 0:
    print("Please provide at least one trajectory")
    exit()

bonds = args.bond
angles = args.angle
dihs = args.dihedral

do_active = args.active
do_pops = args.pop

verb = args.verbose

n_traj = len(trajs)
f = h5py.File(trajs[0], "r")
n_steps = len(f.keys()) - 1
times = np.zeros(n_steps)
for key in f.keys():
    if key == "info":
        n_states = f["info/nst"][()]
        atoms = f["info/ats"][:].astype("<U2")
    else:
        times[int(key)] = f[f"{key}/time"][()]
times = convert(times, "fs")

nout = len(bonds) + len(angles) + len(dihs) + do_active*n_states + do_pops*n_states
data = np.zeros((n_traj, n_steps, nout))
for itraj, traj in enumerate(trajs):
    print(itraj, traj)
    f = h5py.File(traj, "r")
    for key in f.keys():
        if key == "info":
            continue
        step = int(key)
        idx = 0
        pos = f[f"{key}/pos"]

        if do_active:
            act = f[f"{key}/act"][()]
            data[itraj, step, idx+act] = 1
            idx += n_states

        if do_pops:
            pop = np.abs(f[f"{key}/coeff"])**2
            data[itraj, step, idx:idx+n_states] = pop
            idx += n_states

        for bond in bonds:
            r1, r2 = get_pos(pos, *bond)
            res = convert(np.linalg.norm(r1 - r2), "au", "aa")
            data[itraj, step, idx] = res
            idx += 1

        for ang in angles:
            r1, r2, r3 = get_pos(pos, *ang)
            v1 = r1 - r2
            v2 = r3 - r2
            res = np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2))
            data[itraj, step, idx] = res
            idx += 1

        for dih in dihs:
            r1, r2, r3, r4 = get_pos(pos, *dih)
            v1 = r2 - r1
            v2 = r3 - r2
            v3 = r4 - r3
            # https://en.wikipedia.org/wiki/Dihedral_angle
            cr = np.cross(v2, v3)
            res = np.arctan2(np.linalg.norm(v2) * np.dot(v1, cr), np.dot(np.cross(v1, v2), cr))
            data[itraj, step, idx] = res
            idx += 1

if args.output:
    keys = []
    keys.extend([f"popcl_{i}" for i in range(n_states)]*do_active)
    keys.extend([f"popqua_{i}" for i in range(n_states)]*do_pops)
    keys.extend([f"bond_{bond[0]}_{bond[1]}" for bond in bonds])
    keys.extend([f"angle_{angle[0]}_{angle[1]}_{angle[2]}" for angle in angles])
    keys.extend([f"dih_{angle[0]}_{angle[1]}_{angle[2]}_{angle[3]}" for angle in dihs])
    np.savez_compressed(args.output, time = data[0,:,0], **{keys[i]: data[:,:,i] for i in range(nout)})

if args.show:
    idx = 0
    mean = np.mean(data, axis = 0)
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
        ax.plot(times, data[:, :, idx].T, c="r", alpha=5/n_traj)
        if args.mean:
            ax.plot(times, mean[:, idx], c="k")
        ax.set_title(f"{atoms[b1]}{b1}-{atoms[b2]}{b2} Bond Length")
        idx += 1

    for (a1,a2,a3) in angles:
        ax = plt.figure().add_subplot()
        ax.plot(times, data[:, :, idx].T, c="r", alpha=5/n_traj)
        if args.mean:
            ax.plot(times, mean[:, idx], c="k")
        ax.set_title(f"{atoms[a1]}{a1}-{atoms[a2]}{a2}-{atoms[a3]}{a3} Bond Angle")
        idx += 1

    for (a1,a2,a3,a4) in dihs:
        ax = plt.figure().add_subplot()
        ax.plot(times, data[:, :, idx].T, c="r", alpha=5/n_traj)
        if args.mean:
            ax.plot(times, mean[:, idx], c="k")
        ax.set_title(f"{atoms[a1]}{a1}-{atoms[a2]}{a2}-{atoms[a3]}{a3}-{atoms[a4]}{a4} Dihedral Angle")
        idx += 1

    plt.show()