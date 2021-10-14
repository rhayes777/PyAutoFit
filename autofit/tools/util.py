import builtins
import importlib
import json
import os
import re
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Type

import numpy as np


def zip_directory(
        source_directory,
        output=None
):
    output = output or f"{source_directory}.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as f:
        for root, dirs, files in os.walk(source_directory):

            for file in files:
                f.write(
                    os.path.join(root, file),
                    os.path.join(
                        root[len(str(source_directory)):], file
                    ),
                )


def open_(filename, *flags):
    directory = Path(
        filename
    )
    os.makedirs(
        directory.parent,
        exist_ok=True
    )
    return open(filename, *flags)


def get_class_path(cls: type) -> str:
    """
    The full import path of the type
    """
    if hasattr(
        cls,
        "__class_path__"
    ):
        cls = cls.__class_path__
    return re.search("'(.*)'", str(cls))[1]


def get_class(class_path: str) -> Type[object]:
    return GetClass(
        class_path
    ).cls


class GetClass:
    def __init__(self, class_path):
        self.class_path = class_path

    @property
    def _class_path_array(self) -> List[str]:
        """
        A list of strings describing the module and class of the
        real object represented here
        """
        return self.class_path.split(".")

    @property
    def _class_name(self) -> str:
        """
        The name of the real class
        """
        return self._class_path_array[-1]

    @property
    def _module_path(self) -> str:
        """
        The path of the module containing the real class
        """
        return ".".join(self._class_path_array[:-1])

    @property
    def _module(self):
        """
        The module containing the real class
        """
        try:
            return importlib.import_module(
                self._module_path
            )
        except ValueError:
            return builtins

    @property
    def cls(self) -> Type[object]:
        """
        The class of the real object
        """
        return getattr(
            self._module,
            self._class_name
        )


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
        array: np.ndarray, file_path: str, overwrite: bool = False
):
    """
    Write a NumPy array to a json file.

    Parameters
    ----------
    array : np.ndarray
        The array that is written to json.
    file_path : str
        The full path of the file that is output, including the file name and `.json` extension.
    overwrite : bool
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


def numpy_array_from_json(file_path: str):
    """
    Read a 1D NumPy array from a .json file.

    After loading the NumPy array, the array is flipped upside-down using np.flipud. This is so that the structures
    appear the same orientation as .json files loaded in DS9.

    Parameters
    ----------
    file_path : str
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
