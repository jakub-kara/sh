import numpy as np
import os, sys
from typing import List, Callable

class Geo:
    def __init__(self, comm: str, n: int):
        self.x = np.zeros((n, 3))
        self.v = np.zeros((n, 3))
        self.l = np.full(n, "00")
        self.t = 0 #float(comm.replace(" ", "").replace("t=", ""))

class Traj:
    def __init__(self, filename):
        self.g: List[Geo] = []
        ig = -1
        with open(filename, 'r') as file:
            for i, line in enumerate(file):
                if i == 0:
                    n = int(line) + 2
                    continue

                if i % n == 0: continue
                if i % n == 1:
                    self.g.append(Geo(line, n))
                    ig += 1
                    continue

                data = line.strip().split()
                self.g[ig].t = ig
                self.g[ig].l[i%n-2] = data[0]
                self.g[ig].x[i%n-2] = np.array([float(data[1]), float(data[2]), float(data[3])])
                self.g[ig].v[i%n-2] = np.array([float(data[4]), float(data[5]), float(data[6])])

class Command:
    def __init__(self, do: str):
        if do == "dist":
            self.do = dist
        elif do == "rmsd":
            self.do = rmsd
        elif do == "rnoh":
            self.do = rnoh
        elif do == "velo":
            self.do = velo
        elif do == "vdir":
            self.do = vdir
        self.on: List[int] = []

def rmsd(trajs: List[Traj], idx: int, on: List[int]):
    outstr = ""
    for traj in trajs[:-1]:
        d = kabsch(traj.g[idx], trajs[-1].g[idx], rotate=True)
        outstr += f"{d} "
    return outstr

def rnoh(trajs: List[Traj], idx: int, on: List[int]):
    outstr = ""
    for traj in trajs[:-1]:
        d = kabsch(traj.g[idx], trajs[-1].g[idx], rotate=True, noh = True)
        outstr += f"{d} "
    return outstr

def kabsch(geo_in: Geo, ref_in: Geo, rotate: bool = False, noh: bool = False):
    if noh:
        n = geo_in.x.shape[0]
        geox = np.array([geo_in.x[i] for i in range(n) if geo_in.l[i] != "H"])
        geov = np.array([geo_in.v[i] for i in range(n) if geo_in.l[i] != "H"])
        refx = np.array([ref_in.x[i] for i in range(n) if geo_in.l[i] != "H"])
        refv = np.array([ref_in.v[i] for i in range(n) if geo_in.l[i] != "H"])
    else:
        geox = 1*geo_in.x
        geov = 1*geo_in.v
        refx = 1*ref_in.x
        refv = 1*ref_in.v

    n_atoms = geox.shape[0]


    centre = np.sum(refx, axis=0)/n_atoms
    refx -= centre

    #translate
    centre = np.sum(geox, axis=0)/n_atoms
    geox -= centre

    #covariance
    if rotate:
        cov = np.transpose(geox) @ refx
        u, s, v = np.linalg.svd(cov)
        v = np.transpose(v)
        d = np.sign(np.linalg.det(v @ np.transpose(u)))
        f = np.identity(3)
        f[2,2] = d
        r = v @ f @ np.transpose(u)

        #rotation
        for a in range(n_atoms):
            geox[a] = r @ geox[a]

    rmsd = np.sqrt(np.sum((geox - refx)**2 + (geov - refv)**2)/n_atoms)
    return rmsd

def dist(trajs: List[Traj], idx: int, on: List[int]):
    outstr = ""
    for traj in trajs:
        if len(on) == 1:
            d = np.linalg.norm(traj.g[idx].x[on[0]])
        else:
            d = np.linalg.norm(traj.g[idx].x[on[0]] - traj.g[idx].x[on[1]])
        outstr += f"{d} "
    return outstr

def velo():
    pass

def vdir(trajs: List[Traj], idx: int, on: List[int]):
    outstr = ""
    for traj in trajs[:-1]:
        v = traj.g[idx].v/np.linalg.norm(traj.g[idx].v)
        vref = trajs[-1].g[idx].v/np.linalg.norm(trajs[-1].g[idx].v)
        d = np.sum(v*vref)
        outstr += f"{d} "
    return outstr

def main():
    inp = sys.argv[1:]
    flags = ["-i", "-c", "-o"]
    i = 0
    opers: List[Command] = []
    trajs: List[Traj] = []
    while i < len(inp):
        if inp[i] == "-i":
            i += 1

            while inp[i] not in flags:
                trajs.append(Traj(inp[i]))
                i += 1

        if inp[i] == "-c":
            i += 1

            opers.append(Command(inp[i]))
            i += 1

            while inp[i] not in flags:
                opers[-1].on.append(int(inp[i]))
                i += 1

        if inp[i] == "-o":
            i += 1

            outfilename = inp[i]
            i += 1

    with open(outfilename, 'w') as outfile:
        for i, geo in enumerate(trajs[0].g):
            outfile.write(f"{geo.t} ")
            for oper in opers:
                outfile.write(f"{oper.do(trajs, i, oper.on)} ")

            outfile.write("\n")

if __name__ == "__main__":
    main()
