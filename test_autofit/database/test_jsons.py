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
