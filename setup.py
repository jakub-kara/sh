import json
import os, sys
import argparse
import numpy as np

def run_sampling(config: dict):
    print("Sampling")
    samp = config["sampling"]
    n_samp = samp["samples"]
    geos = samp["input"]
    if not geos.endswith(".xyz"):
        distr = {"wigner": 1, "husimi": 2, "ho": 3}[samp.get("distr", "wigner")]
        os.system(f"python3 $SH/sampling/sampler.py -d {distr} -n {n_samp} {'-A'*samp.get('angstrom', False)} {'-a'*samp.get('stats', True)} {geos}")
        geos = "wigner_au.xyz"

    with open(geos, 'r') as f:
        n_atoms = int(f.readline().strip())

    top = int(np.ceil(np.log10(n_samp-1)))
    low = "0"*top
    os.system("rm -r T*")
    os.system(f"mkdir T{{{low}..{n_samp-1}}}")
    os.system(f"split -l {n_atoms + 2} -d --suffix-length={top} {geos} icond_")

    os.system(f"for i in {{{low}..{n_samp-1}}}; do mv icond_$i T$i/geom.xyz; done")
    print("Done")

def run_iconds(config, file):
    print("Initial conditions")
    sub = config.get("run", None)
    com = f"python3 $SH/main.py -i {os.path.basename(file)}"
    if sub:
        sub = "bash " + sub + f" '{com}'"
    else:
        sub = com
    os.system(f"for i in T*; do cp {file} $i; cd $i; mkdir est; {sub}; cd ..; done")
    print("Done")

def run_excite(config, file):
    print("Excitation")
    os.system(f"python3 $SH/sampling/reader.py {file} T*")
    print("Done")

def run_dirs(config, file):
    print("Making directories")
    config["dynamics"]["initstate"] = None
    os.system("rm -r selected")
    os.system("mkdir selected")
    with open("selected.dat", "r") as f:
        f.readline()
        while (line := f.readline()):
            data = line.split()
            orig = data[1]
            config["dynamics"]["initstate"] = int(data[3])
            os.system(f"rm -r {orig}/est")
            os.system(f"mkdir selected/{orig}")
            os.system(f"cp {orig}/geom.xyz selected/{orig}")
            os.system(f"mkdir selected/{orig}/0")
            os.system(f"mkdir selected/{orig}/0/{{backup,data,est}}")
            with open(f"selected/{orig}/{os.path.basename(file)}", "w") as js:
                json.dump(config, js, indent=4)
    print("Done")

def run_trajs(config, file):
    sub = config.get("run", None)
    com = f"python3 $SH/main.py {os.path.basename(file)}"
    if sub:
        sub = "bash " + sub + f" '{com}'"
    else:
        sub = com
    os.system(f"for i in selected/T*; do cd $i; mkdir est; {sub}; cd ..; done")

def main():
    parser = argparse.ArgumentParser(
        prog = "sampling",
        description = "",
        epilog = ""
    )
    parser.add_argument("infile", help="path to input file")
    parser.add_argument("-s", "--sample", action="store_true")
    parser.add_argument("-i", "--iconds", action="store_true")
    parser.add_argument("-e", "--excite", action="store_true")
    parser.add_argument("-d", "--make-dirs", action="store_true")
    parser.add_argument("-r", "--run", action="store_true")
    args = parser.parse_args()

    infile = args.infile
    with open(infile, "r") as file:
        config = json.load(file)

    if args.sample:
        run_sampling(config)

    if args.iconds:
        run_iconds(config, infile)

    if args.excite:
        run_excite(config, infile)

    if args.make_dirs:
        run_dirs(config, infile)

    if args.run:
        run_trajs(config, infile)

if __name__ == "__main__":
    main()