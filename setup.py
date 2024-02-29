import os, sys
sys.path.append(os.path.realpath(__file__))

from io_methods import create_directories, create_geom_files, create_input_files, create_submission_scripts, create_subdirectories, copy_wavefunctions
from utility import get_dirs
from classes import Initialiser

def main():
    path = sys.argv[1].split("/")
    config = Initialiser(path[-1])
    os.chdir("./" + "/".join(path[:-1]))
    if config.ensemble.generate.value:
        create_directories(config)
        create_geom_files(config)
    create_subdirectories(config)
    create_input_files(config)
    create_submission_scripts(config)
    copy_wavefunctions(config)

    orig = os.getcwd()
    for dir in get_dirs(f"{config.general.location.value}/"):
        os.chdir(f"{config.general.location.value}/{dir}")
        os.system(f"{'qsub'*config.general.cluster.value}{'sh'*(not config.general.cluster.value)} submit.sh")
        os.chdir(f"{orig}")

if __name__ == "__main__":
    main()
 