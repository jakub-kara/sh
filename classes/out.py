from pathlib import Path
import h5py
import time
import numpy as np
from classes.config import Config, get_cfg
from classes.constants import convert
from classes.meta import Singleton, Decorator

@Config("filetype")
class BaseWriter:
    """Common functionality for all file writers."""

    def __init__(self, path: Path, **config):
        self.path = path
        self._file = None

    @property
    def is_open(self):
        return self._file is not None

    @property
    def is_closed(self):
        return self._file is None

    def init_file(self):
        self.open(mode = "w")
        self.close()

    def open(self, mode = "a"):
        if self.is_closed:
            self._file = open(self.path, mode = mode)

    def close(self):
        if self.is_open:
            self._file.flush()
            self._file.close()
            self._file = None

    def write(self, *args, **kwargs) -> None:
        raise NotImplementedError

@Config.Option("xyz", "filetype")
class XYZWriter(BaseWriter):
    def write(self, xyzstr: str) -> None:
        self.open()
        self._file.write(xyzstr)

@Config.Option("dat", "filetype")
@Config("record", default=None, options=["ten"], path=Config.OUTPUT, valid=Config.LIST[Config.STR] | Config.NONE)
class DatWriter(BaseWriter):
    def __init__(self, path, **config):
        super().__init__(path)

        record = get_cfg(self, "record", config)
        if isinstance(record, str):
            record = record.split()
        self._record = record


    def write(self, data: dict) -> None:
        self.open()
        file = self._file
        file.write(data["time"])
        for rec in self._record:
            if rec in data.keys():
                file.write(data[rec])
            if rec == "dof":
                for key in data.keys():
                    if key.startswith("x-") or key.startswith("p-"):
                        file.write(data[key])
        file.write("\n")

@Config.Option("log", "filetype")
class LogWriter(BaseWriter):
    def write(self, msg: str = "", newline = True):
        self.open()
        if msg is None:
            return
        self._file.write(f"{msg}" + newline*"\n")

    def pipe(self, filename):
        with open(filename, "r") as f:
            self.write(f.read())

@Config.Option("h5", "filetype")
class H5Writer(BaseWriter):
    def __init__(self, path, **config):
        super().__init__(path)
        self._options = {
            "compression": config.get("compression", "gzip"),
            "compression_opts": config.get("compression_opts", 9),
        }

    def open(self, mode = "a"):
        if self.is_closed:
            self._file = h5py.File(self.path, mode = mode)

    def write(self, data: dict):
        self.open()
        file = self._file
        if data is None:
            return
        key = str(data.pop("step"))
        if key in file:
            del file[key]

        grp = file.create_group(key)
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                grp.create_dataset(key, data=val, **self._options)
            else:
                grp.create_dataset(key, data=val)

# class OutMeta(type):
#     def save(self):
#         return {
#             key: val for key, val in self.__dict__.items()
#             if not (key.startswith("__") and key.endswith("__"))
#             and not callable(val)
#         }

#     def restart(self, dic: dict):
#         for key, val in dic.items():
#             setattr(self, key, val)


@Config("file", default="out", path=Config.OUTPUT, valid=Config.STR)
@Config("log", default=True, path=Config.OUTPUT, valid=Config.BOOL)
@Config("xyz", default=True, path=Config.OUTPUT, valid=Config.BOOL)
@Config("dat", default=True, path=Config.OUTPUT, valid=Config.BOOL)
@Config("h5", default=False, path=Config.OUTPUT, valid=Config.BOOL)
class FileManager:
    files: dict[str, BaseWriter] = {}
    base_dir = Path(".")

    def __init__(self, **config):
        root = get_cfg(self, "file", config)
        for key in ["log", "xyz", "dat", "h5"]:
            if get_cfg(self, key, config):
                self.files[key] = get_cfg(BaseWriter, "filetype", {"filetype": key})(path = self.path_for(root, key), **config)

    @classmethod
    def init_all(cls):
        for file in cls.files.values():
            file.init_file()


    @classmethod
    def open_all(cls):
        for file in cls.files.values():
            file.open()

    @classmethod
    def open(cls, *keys):
        for key in keys:
            if key in cls.files:
                cls.files[key].open()

    @classmethod
    def close(cls, *keys):
        for key in keys:
            if key in cls.files:
                cls.files[key].close()

    @classmethod
    def close_all(cls):
        for file in cls.files.values():
            file.close()

    @classmethod
    def write(cls, key, *args, **kwargs):
        if key in cls.files:
            cls.files[key].write(*args, **kwargs)

    @classmethod
    def log(cls, msg: str = None):
        if msg is not None:
            print(msg)
        cls.write("log", msg)

    @classmethod
    def path_for(cls, name: str, ext: str) -> Path:
        return cls.base_dir / "data" / f"{name}.{ext}"

    def save(self):
        pass

    def restart(self):
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

# class DirChange(Decorator):
#     def __init__(self, id, before, after = None):
#         super().__init__(id)
#         self._before = before
#         if after is None:
#             after = "/".join(".." for i in before.split("/"))
#         self._after = after

#     def run(self, func, instance, *args, **kwargs):
#         os.chdir(self._before)
#         res = func(instance, *args, **kwargs)
#         os.chdir(self._after)
#         return res