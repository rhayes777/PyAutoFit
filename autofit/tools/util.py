import numpy as np
from contextlib import contextmanager
import sys, os
import json

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