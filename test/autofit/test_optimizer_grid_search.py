import pytest

from autofit import mock
from autofit.mapper import model_mapper
from autofit.optimize import grid_search as gs


@pytest.fixture(name="mapper")
def make_mapper():
    mapper = model_mapper.ModelMapper()
    mapper.profile = mock.GeometryProfile
    return mapper


class TestGridSearchablePriors(object):
    def test_generated_models(self, mapper):
        grid_search = gs.GridSearch(model_mapper=mapper, grid_priors=[mapper.profile.centre_0, mapper.profile.centre_1],
                                    step_size=0.1)
        mappers = list(grid_search.models_mappers)

        assert len(mappers) == 100

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 0.1

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.9
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

    def test_non_grid_searched_dimensions(self, mapper):
        grid_search = gs.GridSearch(model_mapper=mapper, grid_priors=[mapper.profile.centre_0], step_size=0.1)
        mappers = list(grid_search.models_mappers)

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 1.0

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.0
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

    def test_tied_priors(self, mapper):
        mapper.profile.centre_0 = mapper.profile.centre_1

        grid_search = gs.GridSearch(model_mapper=mapper, grid_priors=[mapper.profile.centre_0, mapper.profile.centre_1],
                                    step_size=0.1)
        mappers = list(grid_search.models_mappers)

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 0.1

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.9
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

        for mapper in mappers:
            assert mapper.profile.centre_0 == mapper.profile.centre_1
