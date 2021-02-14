import autofit as af
from autofit.mock import mock


class TestMixin:
    def test_mixin(self):
        class MyPhase(af.as_grid_search(af.AbstractPhase)):
            Result = mock.MockResult

            @property
            def grid_priors(self):
                return [self.model.component.one_tuple.one_tuple_0]

            def run(self):
                analysis = mock.MockAnalysis()
                return self.run_analysis(analysis)

        my_phase = MyPhase(
            number_of_steps=2,
            search=mock.MockSearch(
                samples=mock.MockSamples(gaussian_tuples=[(0.5, 0.5), (0.5, 0.5)])
            ),
        )
        my_phase.model.component = mock.MockClassx2Tuple

        result = my_phase.run()

        assert isinstance(result, af.GridSearchResult)
        assert len(result.results) == 2
        assert len(result.lower_limit_lists) == 2
        assert len(result.physical_lower_limits_lists) == 2

        assert isinstance(result.best_result, af.Result)

    def test_parallel_flag(self):
        my_phase = af.as_grid_search(af.AbstractPhase, parallel=True)(
            search=mock.MockSearch()
        )
        assert my_phase.search.parallel
