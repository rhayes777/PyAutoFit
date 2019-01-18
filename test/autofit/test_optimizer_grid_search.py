from autofit import mock
from autofit.mapper import model_mapper
from autofit.optimize import grid_search as gs


class TestGridSearchablePriors(object):
    def test_alternate_models(self):
        mapper = model_mapper.ModelMapper()

        mapper.profile = mock.GeometryProfile

        grid_search = gs.GridSearch(mapper, variables=[mapper.profile.centre_0, mapper.profile.centre_1], step_size=0.1)

        assert len(list(grid_search.models_mappers)) == 100
