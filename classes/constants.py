import numpy as np

is_true = lambda x: x in [True, "true", 1]

atomic_widths = {
    "1": 1, "H": 4.7, "C": 22.7,
}

multiplets = {
    "singlet": 0,
    "doublet": 1,
    "triplet": 2,
    "quartet": 3,
    "quintet": 4,
}

# conversion from au to *keyword*
units = {
    "au": 1,

    # time
    "ps": 2.418884326502e-5,
    "fs": 2.418884326502e-2,
    "as": 2.418884326502e1,

    # distance
    "aa": 5.2917721092e-1,

    # mass
    "amu": 1/1822.8884847700401,

    #energy
    "ev": 2.721138505e1,
    "cm-1": 2.1947463e5,
}

atomic_masses = {
    "1": 1 * units["amu"],
    "2": 2 * units["amu"],
    "H": 1.007825, "HE": 4.003, "LI": 6.941, "BE": 9.012,
    "B": 10.811, "C": 12.00000000, "N": 14.007, "O": 15.999,
    "F": 18.998, "NE": 20.180, "NA": 22.990, "MG": 24.305,
    "AL": 26.982, "SI": 28.086, "P": 30.974, "S": 32.066,
    "CL": 35.453, "AR": 39.948, "K": 39.098, "CA": 40.078,
    "SC": 44.956, "TI": 47.867, "V": 50.942, "CR": 51.996,
    "MN": 54.938, "FE": 55.845, "CO": 58.933, "NI": 58.693,
    "CU": 63.546, "ZN": 65.38, "GA": 69.723, "GE": 72.631,
    "AS": 74.922, "SE": 78.971, "BR": 79.904, "KR": 84.798,
    "RB": 84.468, "SR": 87.62, "Y": 88.906, "ZR": 91.224,
    "NB": 92.906, "MO": 95.95, "TC": 98.907, "RU": 101.07,
    "RH": 102.906, "PD": 106.42, "AG": 107.868, "CD": 112.414,
    "IN": 114.818, "SN": 118.711, "SB": 121.760, "TE": 126.7,
    "I": 126.904, "XE": 131.294, "CS": 132.905, "BA": 137.328,
    "LA": 138.905, "CE": 140.116, "PR": 140.908, "ND": 144.243,
    "PM": 144.913, "SM": 150.36, "EU": 151.964, "GD": 157.25,
    "TB": 158.925, "DY": 162.500, "HO": 164.930, "ER": 167.259,
    "TM": 168.934, "YB": 173.055, "LU": 174.967, "HF": 178.49,
    "TA": 180.948, "W": 183.84, "RE": 186.207, "OS": 190.23,
    "IR": 192.217, "PT": 195.085, "AU": 196.967, "HG": 200.592,
    "TL": 204.383, "PB": 207.2, "BI": 208.980, "PO": 208.982,
    "AT": 209.987, "RN": 222.081, "FR": 223.020, "RA": 226.025,
    "AC": 227.028, "TH": 232.038, "PA": 231.036, "U": 238.029,
    "NP": 237, "PU": 244, "AM": 243, "CM": 247, "BK": 247,
    "CF": 251, "ES": 252, "FM": 257, "MD": 258, "NO": 259,
    "LR": 262, "RF": 261, "DB": 262, "SG": 266, "BH": 264,
    "HS": 269, "MT": 268, "DS": 271, "RG": 272, "CN": 285,
    "NH": 284, "FL": 289, "MC": 288, "LV": 292, "TS": 294,
    "OG": 294}

def convert(val, *args):
    def get_factor(inp: str):
        tot = 1
        conv = inp.split()
        for i in conv:
            spl = i.split("^")
            fact = units[spl[0]]
            if len(spl) > 1:
                fact **= float(spl[1])
            tot *= fact
        return tot

    if len(args) > 2:
        raise ValueError("Too many arguments.")

    unit_in = "au"
    unit_out = args[-1]

    if type(val) == str:
        data = val.split(maxsplit = 1)
        val = float(data[0])
        if len(data) > 1:
            unit_in = data[1]
    elif len(args) == 2:
        unit_in = args[0]

    fact_in = get_factor(unit_in)
    fact_out = get_factor(unit_out)
    return val / fact_in * fact_out
