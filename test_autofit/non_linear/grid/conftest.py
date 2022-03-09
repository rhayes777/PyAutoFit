import pytest

import autofit as af

@pytest.fixture(name="mapper")
def make_mapper():
    return af.Collection(
        component=af.Model(
            af.m.MockClassx2Tuple
        )
    )


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    mock_search = af.m.MockSearch()
    mock_search.paths = af.DirectoryPaths(name="")
    search = af.SearchGridSearch(
        number_of_steps=10,
        search=mock_search
    )
    return search
