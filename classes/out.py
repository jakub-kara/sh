import numpy as np
import pickle, h5py
import time
from classes.meta import Singleton


class Output(metaclass = Singleton):
    def __init__(self, *, file: int, record: list, **config):
        self.record = record
        self._file = file
        self._dist = config.get('dist',False)
        self._options = {
            "compression": config.get("compression", "gzip"),
            "compression_opts": config.get("compression_opts", 9),
        }

        self._log = config.get("log", True)
        self._logfile = None
        self._logmode = "w"

        self._xyz = config.get("xyz", True)
        self._dist = config.get("dist", False)
        self._h5 = config.get("h5", True)
        self._dat = config.get("dat", True)

    def __del__(self):
        self.write_log("TERMINATED")
        self.close_log()

    def to_log(self, filename):
        with open(filename, "r") as f:
            self.write_log(f.read())

    def open_log(self):
        self._logfile = open(f"data/{self._file}.log", self._logmode)
        self._logmode = "a"

    def close_log(self):
        if self._logfile:
            self._logfile.close()
        self._logfile = None

    def write_log(self, msg = ""):
        print(msg)
        if not self._log:
            return
        self._logfile.write(msg + "\n")

    def write_dat(self, data: dict, mode="a"):
        if not self._dat:
            return
        with open(f"data/{self._file}.dat", mode) as file:
            file.write(data["time"])
            for rec in self.record:
                if rec in data.keys():
                    file.write(data[rec])
            file.write("\n")

    def write_dist(self, msg, mode="a"):
        if not self._dist:
            return
        with open(f"data/{self._file}.dist", mode) as file:
            file.write(msg)

    def write_xyz(self, msg, mode="a"):
        if not self._xyz:
            return
        with open(f"data/{self._file}.xyz", mode) as file:
            file.write(msg)

    def write_h5(self, to_write: dict, mode="a"):
        if not self._h5:
            return
        with h5py.File(f"data/{self._file}.h5", mode) as file:
            if to_write is None:
                return
            key = str(to_write["step"])
            if key in file.keys():
                del file[key]
            grp = file.create_group(key)
            to_write.pop("step")
            for key, val in to_write.items():
                if isinstance(val, np.ndarray):
                    grp.create_dataset(key, data=val, **self._options)
                else:
                    grp.create_dataset(key, data=val)

def record_time(fun, out: Output, msg: str = ""):
    def inner(*args, **kwargs):
        t1 = time.time()
        res = fun(*args, **kwargs)
        t2 = time.time()
        out.write_log(f"{msg} {t2-t1}\n")
        return res
    return inner

class Printer:
    field_length = 20
    tdict = {
        "f" : (fform := f" < {field_length}.10e"),
        "p" : (pform := f" < {field_length}.4%"),
        "b" : (bform := f" < 6"),
        "s" : (sform := f" <{field_length}"),
        "i" : (iform := f" < {field_length}.0f"),
    }

    @staticmethod
    def write(val, form):
        if form in Printer.tdict.keys():
            return f"{val:{Printer.tdict[form]}}"
        elif form == "z":
            return f"{val.real:{Printer.tdict['f']}}" + f"{val.imag:< {Printer.field_length-1}.10e}" + "j "
        else:
            return f"{val:{form}}"
