import numpy as np
import pytest

from autofit.database import JSON
from autofit import database as db
from astropy.io import fits


@pytest.fixture(name="fit")
def make_fit():
    return db.Fit()


def test_json():
    json = JSON(name="test", dict={"a": 1})
    assert json.name == "test"
    assert json.dict == {"a": 1}


def test_set_json(fit):
    fit.set_json("test", {"a": 1})
    assert fit.get_json("test") == {"a": 1}


def test_array():
    csv = db.Array(
        name="test",
        array=np.array([[1, 2], [3, 4]]),
    )
    assert csv.name == "test"
    assert (csv.array == [[1, 2], [3, 4]]).all()


def test_set_array(fit):
    fit.set_array("test", np.array([[1, 2], [3, 4]]))
    assert (fit.get_array("test") == [[1, 2], [3, 4]]).all()


@pytest.fixture(name="hdu_array")
def make_hdu_array():
    return np.array([[3, 3], [3, 3]], dtype=np.dtype(">f8"))


@pytest.fixture(name="hdu")
def make_hdu(hdu_array):
    new_hdr = fits.Header()
    return fits.PrimaryHDU(hdu_array, new_hdr)


def test_hdu(hdu, hdu_array):
    db_hdu = db.HDU(name="test", hdu=hdu)
    assert db_hdu.name == "test"

    loaded = db_hdu.hdu
    assert (loaded.data == hdu_array).all()
    assert loaded.header == hdu.header


def test_set_hdu(fit, hdu, hdu_array):
    fit.set_hdu("test", hdu)
    loaded = fit.get_hdu("test")
    assert (loaded.data == hdu_array).all()
    assert loaded.header == hdu.header
