import pytest

import autofit as af
from autofit.mock import mock


@pytest.fixture(name="mapper")
def make_mapper():
    return af.Collection(
        component=af.Model(
            mock.MockClassx2Tuple
        )
    )


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    search = af.SearchGridSearch(
        number_of_steps=10, search=af.MockSearch()
    )
    search.paths = af.DirectoryPaths(name="")
    return search
