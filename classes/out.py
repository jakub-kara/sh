import numpy as np
import h5py
import time
import atexit
import os
from classes.meta import Decorator

class OutMeta(type):
    border = "="*40

    def setup(self, *, file: str = "out", record: list = "", **config):
        if not isinstance(record, list):
            record = record.split()
        self._record = record
        self._file = file
        self._dist = config.get("dist", False)
        self._options = {
            "compression": config.get("compression", "gzip"),
            "compression_opts": config.get("compression_opts", 9),
        }

        self._log = config.get("log", True)
        self._logfile = None
        self._xyz = config.get("xyz", True)
        self._dist = config.get("dist", False)
        self._h5 = config.get("h5", True)
        self._dat = config.get("dat", True)
        if self._record == []:
            self._dat = False

    def to_log(self, filename):
        with open(filename, "r") as f:
            self.write_log(f.read())

    def open_log(self, mode = "a"):
        self._logfile = open(f"data/{self._file}.log", mode = mode)

    def close_log(self):
        if self._logfile:
            self._logfile.close()
        self._logfile = None

    def write_border(self):
        self.write_log(self.border)

    def write_log(self, msg = "", mode = "a"):
        if msg is None:
            return
        if not self._log:
            return
        print(msg)
        with open(f"data/{self._file}.log", mode) as file:
            file.write(f"{msg}\n")

    def write_dat(self, data: dict, mode="a"):
        if not self._dat:
            return
        with open(f"data/{self._file}.dat", mode) as file:
            file.write(data["time"])
            for rec in self._record:
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

    def save(self):
        return {
            key: val for key, val in self.__dict__.items()
            if not (key.startswith("__") and key.endswith("__"))
            and not callable(val)
        }

    def restart(self, dic: dict):
        for key, val in dic.items():
            setattr(self, key, val)

class Output(metaclass = OutMeta):
    pass

class Timer(Decorator):
    def __init__(self, id, head = "", msg = "Wall time: ", foot = "", out = lambda x: None):
        super().__init__(id)
        self._head = head
        self._msg = msg
        self._foot = foot
        self._out = out

    def run(self, func, instance, *args, **kwargs):
        self._out(f"{self._head}")
        t0 = time.time()
        res = func(instance, *args, **kwargs)
        self._out(f"{self._msg}: {time.time() - t0 :.4f} s")
        self._out(f"{self._foot}")
        return res

class DirChange(Decorator):
    def __init__(self, id, before, after = None):
        super().__init__(id)
        self._before = before
        if after is None:
            after = "/".join(".." for i in before.split("/"))
        self._after = after

    def run(self, func, instance, *args, **kwargs):
        os.chdir(self._before)
        res = func(instance, *args, **kwargs)
        os.chdir(self._after)
        return res

class Logger(Decorator):
    def run(self, func, instance, *args, **kwargs):
        Output.open_log()
        res = func(instance, *args, **kwargs)
        Output.close_log()
        return res

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
