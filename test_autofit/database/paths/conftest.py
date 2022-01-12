import pytest

import autofit as af


@pytest.fixture(
    name="paths"
)
def make_paths(session):
    paths = af.DatabasePaths(
        session
    )
    paths.model = af.Model(
        af.Gaussian
    )
    assert paths.is_complete is False
    return paths
