import pytest

import autofit as af
from autofit.mock.mock_model import MockClassx2Tuple
from autofit.mock.mock import MockSearch

@pytest.fixture(name="mapper")
def make_mapper():
    return af.Collection(
        component=af.Model(
            MockClassx2Tuple
        )
    )


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    mock_search = MockSearch()
    mock_search.paths = af.DirectoryPaths(name="")
    search = af.SearchGridSearch(
        number_of_steps=10,
        search=mock_search
    )
    return search
