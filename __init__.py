import os, sys
import importlib
from pathlib import Path

path = Path(__file__).parent.absolute()
include = ["dynamics", "electronic", "updaters"]

for dirpath, dirnames, filenames in os.walk(path):
    dirpath = os.path.relpath(dirpath, path)
    if not any(map(dirpath.startswith, include)):
        continue

    for file in filenames:
        if file.endswith(".py") and not file.startswith("__"):
            imp = dirpath.replace("/", ".") + "." + file[:-3]
            # print(f"Importing {imp}")
            importlib.import_module(imp)
