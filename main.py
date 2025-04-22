import imports
import os, json
import time
import argparse

from sampling.icond import icond
from classes.bundle import Bundle
from dynamics.base import Dynamics

def main():
    parser = argparse.ArgumentParser(
        prog = "SHREC",
        description = "Main script to run SHREC",
        epilog = ""
    )
    parser.add_argument("infile", help="path to input file")
    parser.add_argument("-r", "--restart", action="store_true")
    parser.add_argument("-i", "--icond", action="store_true")
    args = parser.parse_args()

    t_ini = time.time()

    with open(args.infile, "r") as file:
        config = json.load(file)
    if "quantum" not in config.keys():
        config["quantum"] = {}
    if "nuclear" not in config.keys():
        config["nuclear"] = {}

    if args.icond:
        run_icond(args, config)
    else:
        run_dynamics(args, config)

    t_fin = time.time()
    print()
    print("=======================================")
    print(f"Total time: {t_fin - t_ini}")
    print("=======================================")

def run_icond(args, config):
    icond(**config)

def run_dynamics(args, config: dict):
    bundle: Bundle
    if args.restart:
        bundle = Bundle.restart(**config)
    else:
        Dynamics.set_dynamics(**config)
        bundle = Bundle(**config)
        Dynamics().prepare_bundle(bundle)

    while not bundle.is_finished:
        Dynamics().step_bundle(bundle)

if __name__ == "__main__":
    main()
