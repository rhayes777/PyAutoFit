import pytest

from autofit import SearchOutput
from autofit.database.aggregator.scrape import _add_files


class MockFit:
    def __init__(self):
        self.jsons = {}

    id = "id"

    def set_json(self, name, json):
        self.jsons[name] = json

    def __setitem__(self, key, value):
        pass

    def set_pickle(self, key, value):
        pass

    def set_array(self, key, value):
        pass

    def set_fits(self, key, value):
        pass


@pytest.fixture(name="fit")
def make_fit(directory):
    fit = MockFit()
    _add_files(
        fit=fit,
        item=SearchOutput(directory / "search_output"),
    )
    return fit


def test_add_files(fit):
    assert fit.jsons["model"] == {
        "class_path": "autofit.example.model.Gaussian",
        "type": "model",
        "arguments": {
            "centre": {
                "lower_limit": "-inf",
                "upper_limit": "inf",
                "type": "Gaussian",
                "id": 0,
                "mean": 1.0,
                "sigma": 1.0,
            },
            "normalization": {
                "lower_limit": "-inf",
                "upper_limit": "inf",
                "type": "Gaussian",
                "id": 1,
                "mean": 1.0,
                "sigma": 1.0,
            },
            "sigma": {
                "lower_limit": "-inf",
                "upper_limit": "inf",
                "type": "Gaussian",
                "id": 2,
                "mean": 1.0,
                "sigma": 1.0,
            },
        },
    }


def test_add_recursive(fit):
    assert fit.jsons["directory.example"] == {
        "hello": "world",
    }
