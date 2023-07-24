import numpy as np

from autoconf.class_path import get_class_path, get_class
from autofit.database import JSON
from autofit import database as db


def test_json():
    json = JSON(name="test", dict={"a": 1})
    assert json.name == "test"
    assert json.dict == {"a": 1}


def test_set_json():
    fit = db.Fit()
    fit.set_json("test", {"a": 1})
    assert fit.get_json("test") == {"a": 1}


def test_array():
    csv = db.Array(
        name="test",
        array=np.array([[1, 2], [3, 4]]),
    )
    assert csv.name == "test"
    print(csv.array)
    assert (csv.array == [[1, 2], [3, 4]]).all()
