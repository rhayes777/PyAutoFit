from os import path

import pytest

import autofit as af


@pytest.fixture(name="aggregator")
def make_path_aggregator(session):
    fits = [
        af.db.Fit(
            id="complete",
            is_complete=True
        ),
        af.db.Fit(
            id="incomplete",
            is_complete=False
        )
    ]
    for i, fit in enumerate(fits):
        fit["dataset"] = {
            "name": "dataset"
        }
        fit["pipeline"] = f"pipeline{i}"
    session.add_all(fits)
    session.flush()
    return af.Aggregator(session)


@pytest.fixture(name="aggregator_directory")
def make_aggregator_directory():
    directory = path.dirname(path.realpath(__file__))

    return path.join(f"{directory}", "..", "tools", "files", "aggregator")
