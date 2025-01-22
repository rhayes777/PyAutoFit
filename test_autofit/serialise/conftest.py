import itertools

import pytest
import autofit as af


@pytest.fixture(autouse=True)
def reset_ids():
    af.Prior._ids = itertools.count()
