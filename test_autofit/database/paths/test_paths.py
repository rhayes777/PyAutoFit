import pytest

import autofit as af
from autofit import database as m


@pytest.fixture(
    name="paths"
)
def make_paths(session):
    paths = af.DatabasePaths(
        session
    )
    assert paths.is_complete is False
    return paths


@pytest.fixture(
    name="fit"
)
def query_fit(session, paths):
    fit, = m.Fit.all(session)
    return fit


def test_identifier(
        paths,
        fit
):
    assert fit.id == paths.identifier


def test_completion(
        paths,
        fit
):
    paths.completed()

    assert fit.is_complete
    assert paths.is_complete
