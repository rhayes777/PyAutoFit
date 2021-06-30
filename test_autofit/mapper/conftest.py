import pytest

import autofit as af


@pytest.fixture(name="mapper")
def make_mapper():
    return af.ModelMapper()
