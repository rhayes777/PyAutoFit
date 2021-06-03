import pytest

import autofit as af
from autofit.mock.mock import Gaussian


@pytest.fixture(
    name="paths"
)
def make_paths(session):
    paths = af.DatabasePaths(
        session
    )
    paths.model = af.Model(
        Gaussian
    )
    assert paths.is_complete is False
    return paths
