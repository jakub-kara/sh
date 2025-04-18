import numpy as np
import pickle, h5py
import time
from classes.meta import Singleton, Decorator

class Output(metaclass = Singleton):
    border = "="*40

    def __init__(self, *, file: str = "out", record: list = "", **config):
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

    def __del__(self):
        self.write_log()
        self.write_border()
        self.write_log("Program terminated.")
        self.write_log("Exiting.")
        self.write_border()
        self.close_log()

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

    def write_log(self, msg = ""):
        if msg is None:
            return
        if not self._log:
            return
        print(msg)
        self._logfile.write(f"{msg}\n")

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

class Logger(Decorator):
    def run(self, func, instance, *args, **kwargs):
        print(f"Running method {func} of {instance}")
        res = func(instance, *args, **kwargs)
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
