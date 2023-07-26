import pytest

from autofit.database.aggregator.scrape import _add_files


class MockFit:
    def __init__(self):
        self.jsons = {}

    def set_json(self, name, json):
        self.jsons[name] = json


@pytest.fixture(name="fit")
def make_fit(directory):
    fit = MockFit()
    _add_files(
        fit=fit,
        files_path=directory / "search_output" / "files",
    )
    return fit


def test_add_files(fit):
    assert fit.jsons["model"] == {
        "centre": {
            "id": 0,
            "lower_limit": "-inf",
            "mean": 1.0,
            "sigma": 1.0,
            "type": "Gaussian",
            "upper_limit": "inf",
        },
        "class_path": "autofit.example.model.Gaussian",
        "normalization": {
            "id": 1,
            "lower_limit": "-inf",
            "mean": 1.0,
            "sigma": 1.0,
            "type": "Gaussian",
            "upper_limit": "inf",
        },
        "sigma": {
            "id": 2,
            "lower_limit": "-inf",
            "mean": 1.0,
            "sigma": 1.0,
            "type": "Gaussian",
            "upper_limit": "inf",
        },
        "type": "model",
    }


def test_add_recursive(fit):
    assert fit.jsons["directory.example"] == {
        "hello": "world",
    }
