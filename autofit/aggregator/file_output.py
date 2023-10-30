import json
from abc import abstractmethod, ABC
from pathlib import Path

import numpy as np

import dill

from autoconf.dictable import from_dict


class FileOutput(ABC):
    def __new__(cls, path: Path):
        suffix = path.suffix
        if suffix == ".pickle":
            return super().__new__(PickleOutput)
        elif suffix == ".json":
            return super().__new__(JSONOutput)
        elif suffix == ".csv":
            return super().__new__(CSVOutput)
        raise ValueError(f"File {path} is not a valid output file")

    def __init__(self, path: Path):
        self.path = path

    @property
    def name(self):
        return self.path.stem

    @property
    @abstractmethod
    def value(self):
        pass


class CSVOutput(FileOutput):
    @property
    def value(self):
        return np.loadtxt(self.path, delimiter=",")


class JSONOutput(FileOutput):
    @property
    def dict(self):
        with open(self.path) as f:
            return json.load(f)

    @property
    def value(self):
        return from_dict(self.dict)


class PickleOutput(FileOutput):
    @property
    def value(self):
        with open(self.path, "rb") as f:
            return dill.load(f)


class FitsOutput(FileOutput):
    @property
    def value(self):
        from astropy.io import fits

        with open(self.path, "rb") as f:
            return fits.PrimaryHDU.readfrom(f)
