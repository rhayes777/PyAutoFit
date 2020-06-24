import autofit as af
from test_autofit.unit.mapper.model.test_model_mapper import GeometryProfile


class TestMixin:
    def test_mixin(self, container):
        class MyPhase(af.as_grid_search(af.AbstractPhase)):
            @property
            def grid_priors(self):
                return [self.model.profile.centre_0]

            def run(self):
                analysis = container.MockAnalysis()
                return self.make_result(self.run_analysis(analysis), analysis)

        my_phase = MyPhase(
            af.Paths(name="", folders=tuple()),
            number_of_steps=2,
            search=container.MockOptimizer,
        )
        my_phase.model.profile = GeometryProfile

        result = my_phase.run()

        assert isinstance(result, af.GridSearchResult)
        assert len(result.results) == 2
        assert len(result.lower_limit_lists) == 2
        assert len(result.physical_lower_limits_lists) == 2

        assert isinstance(result.best_result, af.Result)

    def test_parallel_flag(self):
        my_phase = af.as_grid_search(af.AbstractPhase, parallel=True)(
            af.Paths(name="phase name")
        )
        assert my_phase.search.parallel
