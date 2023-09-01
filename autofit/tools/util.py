import inspect
import json
import os
import sys
import zipfile
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Union

import numpy as np

from autoconf import conf
from autoconf.class_path import get_class_path, get_class
from autofit.mapper.model import ModelObject


def to_dict(obj):
    """
    Convert an object to a dictionary.

    The dictionary can be converted back to the object using `from_dict`.

    The representation is recursive, so dictionaries and lists are also converted.
    A type describes the path to the class of the object, and arguments maps
    constructor arguments to values of attributes with the same name.
    """
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if inspect.isclass(obj):
        return {
            "type": "type",
            "class_path": get_class_path(obj),
        }
    if isinstance(obj, dict):
        return {
            "type": "dict",
            "arguments": {
                ".".join(key) if isinstance(key, tuple) else key: to_dict(value)
                for key, value in obj.items()
            },
        }
    if isinstance(obj, list):
        return {
            "type": "list",
            "values": [to_dict(value) for value in obj],
        }
    if inspect.isclass(type(obj)):
        arguments = set(inspect.getfullargspec(obj.__init__).args[1:])
        try:
            arguments |= set(obj.__identifier_fields__)
        except (AttributeError, TypeError):
            pass
        return {
            "type": "instance",
            "class_path": get_class_path(type(obj)),
            "arguments": {
                argument: to_dict(getattr(obj, argument))
                for argument in arguments
                if hasattr(obj, argument)
            },
        }
    return obj


def from_dict(dictionary):
    """
    Convert a dictionary to an object.
    """
    if isinstance(dictionary, (int, float, str, bool, type(None))):
        return dictionary

    type_ = dictionary["type"]
    if type_ == "type":
        return get_class(dictionary["class_path"])
    if type_ == "instance":
        cls = get_class(dictionary["class_path"])
        return cls(
            **{
                argument: from_dict(value)
                for argument, value in dictionary["arguments"].items()
            }
        )

    if type_ == "list":
        return [from_dict(value) for value in dictionary["values"]]
    if type_ == "dict":
        return {key: from_dict(value) for key, value in dictionary["arguments"].items()}

    if type_ in (
        "model",
        "collection",
        "tuple_prior",
        "dict",
        "instance",
    ):
        return ModelObject.from_dict(dictionary)
    cls = get_class(type_)
    if hasattr(cls, "from_dict"):
        return cls.from_dict(dictionary)
    raise ValueError(f"Cannot convert {dictionary} to an object")


def split_paths(func):
    """
    Split string paths if they are passed.

    e.g. "lens.mass.centre" -> ["lens", "mass", "centre"]
    """

    @wraps(func)
    def wrapper(self, paths):
        paths = [path.split(".") if isinstance(path, str) else path for path in paths]
        return func(self, paths)

    return wrapper


class IntervalCounter:
    def __init__(self, interval):
        self.count = 0
        self.interval = interval

    def __call__(self):
        if self.interval == -1:
            return False
        self.count += 1
        return self.count % self.interval == 0


def zip_directory(source_directory, output=None):
    output = output or f"{source_directory}.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as f:
        for root, dirs, files in os.walk(source_directory):
            for file in files:
                f.write(
                    os.path.join(root, file),
                    os.path.join(root[len(str(source_directory)) :], file),
                )


def open_(filename, *flags):
    directory = Path(filename)
    os.makedirs(directory.parent, exist_ok=True)
    return open(filename, *flags)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def numpy_array_to_json(
    array: np.ndarray, file_path: Union[Path, str], overwrite: bool = False
):
    """
    Write a NumPy array to a json file.

    Parameters
    ----------
    array
        The array that is written to json.
    file_path
        The full path of the file that is output, including the file name and `.json` extension.
    overwrite
        If `True` and a file already exists with the input file_path the .json file is overwritten. If
        `False`, an error will be raised.

    Returns
    -------
    None

    Examples
    --------
    array_2d = np.ones((5,5))
    numpy_array_to_json(array_2d=array_2d, file_path='/path/to/file/filename.json', overwrite=True)
    """

    file_dir = os.path.split(file_path)[0]

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w+") as f:
        json.dump(array.tolist(), f)


def numpy_array_from_json(file_path: Union[Path, str]):
    """
    Read a 1D NumPy array from a .json file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures
    appear the same orientation as .json files loaded in DS9.

    Parameters
    ----------
    file_path
        The full path of the file that is loaded, including the file name and ``.json`` extension.

    Returns
    -------
    ndarray
        The NumPy array that is loaded from the .json file.

    Examples
    --------
    array_2d = numpy_array_from_json(file_path='/path/to/file/filename.json')
    """
    with open(file_path, "r") as f:
        return np.asarray(json.load(f))


def info_whitespace():
    return conf.instance["general"]["output"]["info_whitespace_length"]
