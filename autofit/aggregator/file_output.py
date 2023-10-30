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

    def __init__(self, file_path: Path):
        self.file_path = file_path

    @property
    def name(self):
        return self.file_path.stem

    @property
    @abstractmethod
    def value(self):
        pass


class CSVOutput(FileOutput):
    @property
    def value(self):
        return np.loadtxt(self.file_path, delimiter=",")


class JSONOutput(FileOutput):
    @property
    def dict(self):
        with open(self.file_path) as f:
            return json.load(f)

    @property
    def value(self):
        return from_dict(self.dict)


class PickleOutput(FileOutput):
    @property
    def value(self):
        with open(self.file_path, "rb") as f:
            return dill.load(f)
