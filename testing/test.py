import numpy as np
import sys, json
from dynamics import select_dynamics
from classes.bundle import Bundle

if len(sys.argv) > 1:
    input = sys.argv[1]
    with open(input, "r") as file:
        config = json.load(file)
    bundle = Bundle().setup(config)
else:
    bundle = Bundle.from_pkl()
while not bundle.is_finished:
    bundle.run_step()