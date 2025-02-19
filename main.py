import __init__
import sys, json
import time
import argparse

from classes.bundle import Bundle


parser = argparse.ArgumentParser(
    prog = "SHREC",
    description = "Main script to run SHREC",
    epilog = ""
)
parser.add_argument("infile", help="path to input file")
parser.add_argument("-r", "--restart", action="store_true")
args = parser.parse_args()

t_ini = time.time()
input = args.infile
with open(input, "r") as file:
    config = json.load(file)

bundle: Bundle
if args.restart:
    bundle = Bundle.restart(**config)
else:
    bundle = Bundle().setup(**config)

while not bundle.is_finished:
    bundle.run_step()

t_fin = time.time()
print()
print("=======================================")
print(f"Total time: {t_fin - t_ini}")
print("=======================================")
