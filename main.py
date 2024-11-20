import __init__
import sys, json
from classes.bundle import Bundle
import time


t_ini = time.time()
if len(sys.argv) > 1:
    input = sys.argv[1]
    with open(input, "r") as file:
        config = json.load(file)
    bundle = Bundle().setup(config)
else:
    bundle = Bundle.from_pkl()

while not bundle.is_finished:
    bundle.run_step()

t_fin = time.time()
print()
print("=======================================")
print(f"Total time: {t_fin - t_ini}")
print("=======================================")
