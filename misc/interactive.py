import json
import os, sys
import readline

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}

EMPTY = [None, -1, "", [], [""]]

def validate_input(expr):
    try:
        return expr()
    except KeyboardInterrupt:
        exit()
    except:
        return None

def set_value(expr, default=None, defmsg=None):
    if default is None:
        temp = None
        while temp in EMPTY:
            while (temp:=validate_input(expr)) is None: pass
        return temp
    else:
        while (temp:=validate_input(expr)) is None: pass
        if temp in EMPTY:
            print(defmsg or default)
            return default
        else:
            return temp

def main():
    config = {}

    os.system("clear")
    print("Interactive setup script for SH")
    print("")

    print("(1) Control settings")
    config["control"] = {}
    print("Trajectory type [sh]")
    config["control"]["type"] = set_value(lambda: input().lower(), "sh")
    print("Path to ensemble [.]")
    config["control"]["location"] = set_value(lambda: input(), ".")
    print("Name of molecule [x]")
    config["control"]["name"] = set_value(lambda: input(), "x")
    print("Quantities to record (space-separated) [pes]")
    config["control"]["record"] = set_value(lambda: input().lower().split(), ["pes"], "pes")

    print("Input units [au]")
    config["control"]["tunit"] = set_value(lambda: input().lower(), "au")
    print("Total time")
    config["control"]["tmax"] = set_value(lambda: float(input()))
    print("Adaptive stepsize [n]")
    config["control"]["adapt"] = set_value(lambda: input() in Constants.true, False, "n")
    if config["control"]["adapt"]:
        print("Stepsize function [tanh]")
        config["control"]["stepfunc"] = set_value(lambda: input().lower(), "tanh")
        print("Stepsize input variable [nac**2]")
        config["control"]["stepvar"] = set_value(lambda: input().lower(), "nac2")
        print("Max stepsize")
        config["control"]["stepmax"] = set_value(lambda: float(input()))
        print("Min stepsize")
        config["control"]["stepmin"] = set_value(lambda: float(input()))
        print("Stepsize parameters (space-separated)")
        config["control"]["stepparams"] = set_value(lambda: [float(i) for i in input().lower().split()])
    else:
        print("Stepsize")
        config["control"]["stepmax"] = set_value(lambda: float(input()))
        config["control"]["stepmin"] = config["control"]["stepmax"]
        config["control"]["stepfunc"] = "const"
        config["control"]["stepparams"] = []
    print("Number of quantum substeps [20]")
    config["control"]["qres"] = set_value(lambda: int(input() or -1), 20)
    print("")

    print("(2) Nuclear settings")
    config["nuclear"] = {}
    print("Input format [xyz]")
    config["nuclear"]["format"] = set_value(lambda: input(), "xyz")
    print("Nuclear integrator [vv]")
    config["nuclear"]["integrator"] = set_value(lambda: input().lower(), "vv")
    print("")

    print("(3) Electronic structure settings")
    config["electronic"] = {}
    print("Number of states")
    config["electronic"]["nstates"] = set_value(lambda: int(input() or -1))
    print("Initial state (0-based indexing)")
    config["electronic"]["initstate"] = set_value(lambda: int(input() or -1))
    print("Skip states [0]")
    config["electronic"]["skip"] = set_value(lambda: int(input() or -1), 0)
    print("Wavefunction coefficients propagator [propmat]")
    config["electronic"]["propmat"] = set_value(lambda: input().lower(), "propmat")
    print("EST Program [molpro]")
    config["electronic"]["program"] = set_value(lambda: input().lower(), "molpro")
    if config["electronic"]["program"] != "model":
        print(f"Path to {config['electronic']['program'].upper()} [{config['electronic']['program'].upper()}]")
        config["electronic"]["programpath"] = set_value(lambda: input(), config['electronic']['program'].upper())
        if config["electronic"]["programpath"] in os.environ.keys(): config["electronic"]["programpath"] = os.environ(config["electronic"]["programpath"])
        print("Initial wavefunction [wf.wf]")
        config["electronic"]["wf"] = set_value(lambda: input(), "wf.wf")
        print("Calculation settings")
        config["electronic"]["config"] = {}
        print("Number of electrons")
        config["electronic"]["config"]["nel"] = set_value(lambda: int(input() or -1))
        print("Closed orbitals")
        config["electronic"]["config"]["closed"] = set_value(lambda: int(input() or -1))
        print("Active orbitals")
        config["electronic"]["config"]["active"] = set_value(lambda: int(input() or -1))
        print("Basis")
        config["electronic"]["config"]["basis"] = set_value(lambda: input())
        print(f"State Average [{config['electronic']['nstates']}]")
        config["electronic"]["config"]["sa"] = set_value(lambda: int(input() or -1), config["electronic"]["nstates"])
        print("Density fitting [n]")
        config["electronic"]["config"]["df"] = set_value(lambda: input() in Constants.true, False)
        if config["electronic"]["config"]["df"]:
            print("Basis for density fitting [avdz]")
            config["electronic"]["config"]["dfbasis"] = set_value(lambda: input(), "avdz")
        print("Save molden every x steps [never]")
        config["electronic"]["config"]["mld"] = set_value(lambda: int(input() or -1), -1, f"{COLOR['RED']}never{COLOR['ENDC']}")
    else:
        print("Model type")
        config["electronic"]["type"] = set_value(lambda: input().lower())
    print("")

    if config["control"]["type"] == "sh":
        print("(4) Surface hopping")
        config["hopping"] = {}
        print("Hopping type [fssh]")
        config["hopping"]["type"] = set_value(lambda: input().lower(), "fssh")
        print("Decoherence [edc]")
        config["hopping"]["decoherence"] = set_value(lambda: input().lower(), "edc")

    with open("input.json", "w") as inp:
        json.dump(config, inp, indent=4)


if __name__  == "__main__":
    main()