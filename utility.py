import os
import numpy as np

from constants import Constants

def file_to_dictionary(file_name: str):
    with open(file_name, "r") as file:
        out_dict = {}
        sub_dict = ""
        while True:
            line = file.readline()
            if not line:
                break
            
            line = line.strip().replace(" ", "")
            if line.startswith("!") or line == "": 
                continue

            data = line.split("!",1)[0].split("=", 1)
            if len(data) == 1:
                out_dict[data[0]] = {}
                sub_dict = data[0]
                continue
            
            val = data[1]
            if data[1].startswith("{"):
                while True:
                    line = file.readline()
                    if not line:
                        raise Exception("Unpaired brackets in input file.")
                    val += line
                    if "}" in line.split("!")[0]:
                        break
            out_dict[sub_dict][data[0]] = val
    return out_dict

def get_dict_value(dict: dict, default = "mandatory", *keys):
    if len(keys) > 1:
        try:
            return get_dict_value(dict[keys[0]], default, *keys[1:])
        except KeyError:
            return default
    else:
        if keys[0] in dict:
            return dict[keys[0]]
        elif default == "mandatory":
            raise Exception(f"{keys[0]} is a mandatory input parameter.")
        else:
            return default        

def get_dirs(path="."):
    return [f for f in os.listdir(path) if os.path.isdir(f)]

def get_ext(ext, path="."):
    return [f for f in os.listdir(path) if os.path.isfile(f) and f.endswith(f".{ext}")]

def read_initial_conditions(input_file):
    with open(input_file, 'r') as open_file:
        for i, line in enumerate(open_file):
            if i == 0:
                assert len(line.split()) == 1
                n_atoms = int(line)
                atom_name_a = np.full(n_atoms, "00")
                position_ad = np.zeros((n_atoms, 3), order='F')
                velocity_ad = np.zeros((n_atoms, 3), order='F')
                mass_a = np.zeros(n_atoms)
            elif i == 1:
                comment = line
            else:
                line_list = line.split()
                if len(line_list) > 0:
                    assert len(line_list) == 7, "wrong xyz file format"
                    atom_name_a[i-2] = line_list[0]
                    mass_a[i-2] = Constants.atomic_masses[line_list[0]]*Constants.amu
                    position_ad[i-2,:] = [float(num.replace('d', 'e')) for num in line_list[1:4]]
                    velocity_ad[i-2,:] = [float(num.replace('d', 'e')) for num in line_list[4:7]]

    return n_atoms, position_ad, velocity_ad, atom_name_a, mass_a