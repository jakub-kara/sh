import os, sys

from io_methods import create_directories, create_geom_files, create_input_files, create_submission_scripts, create_subdirectories
from utility import file_to_dictionary, get_dict_value, get_dirs
from constants import Constants

def main():
    input_dict = file_to_dictionary(sys.argv[1])
    os.chdir("./" + "/".join(sys.argv[1].split("/")[:-1]))
    if get_dict_value(input_dict["ensemble"], "generate", "true") in Constants.true:
        create_directories(input_dict)
        create_geom_files(input_dict)
    create_input_files(input_dict)
    create_submission_scripts(input_dict)

    location = get_dict_value(input_dict["general"], "location", ".")
    cluster = get_dict_value(input_dict["general"], "cluster", "false") in Constants.true
    orig = os.getcwd()
    for dir in get_dirs(f"{location}/"):
        os.chdir(f"{location}/{dir}/")
        create_subdirectories()
        os.system(f"{'qsub '*cluster}{'sh '*(not cluster)}./submit.sh")
        os.chdir(f"{orig}/")

if __name__ == "__main__":
    main()
