"""
The EP adapter of a non-linear search (`SearchFactorOptimiser`, search-extensibility
A4): a 2-step EP run with a MockSearch.

`NonLinearSearch.optimise` delegates to the adapter, which keeps the semantics it had
on the search: one `optimization_<n>` sub-folder per factor per EP step (a fresh
paths object each time), the cached updater rebuilt for each new paths object, the
per-factor visual switches restored after each step, and the in-memory search
internal released before the result is pinned on the `Status` (#1631).
"""

import pytest

import autofit as af
from autofit import graphical as g
from autofit.graphical.expectation_propagation import (
    AbstractFactorOptimiser,
    SearchFactorOptimiser,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

MAX_STEPS = 2


class _Result(af.m.MockResult):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.released = False

    @property
    def projected_model(self):
        return self.model

    def release_search_internal(self):
        self.released = True


class _RecordingSearch(af.m.MockSearch):
    """A MockSearch whose `fit` records what EP set up for each factor step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = []
        self.results = []

    def fit(self, model, analysis, info=None):
        updater = self._updater
        self.steps.append(
            dict(
                analysis_name=self.paths.analysis_name,
                paths=self.paths,
                updater=updater,
                updater_paths_current=updater._paths is self.paths,
                visualize_fit=self._visualize_fit,
                visualize_before_fit=self._visualize_before_fit,
            )
        )
        result = _Result(model)
        self.results.append(result)
        return result


@pytest.fixture(name="factor_graph_model")
def make_factor_graph_model():
    return g.FactorGraphModel(
        g.AnalysisFactor(af.Collection(one=af.UniformPrior()), af.m.MockAnalysis()),
        g.AnalysisFactor(af.Collection(one=af.UniformPrior()), af.m.MockAnalysis()),
    )


@pytest.fixture(name="search")
def make_search(factor_graph_model, tmp_path):
    search = _RecordingSearch()
    factor_graph_model.optimise(
        search,
        max_steps=MAX_STEPS,
        paths=af.DirectoryPaths(name="ep_adapter", path_prefix=str(tmp_path)),
    )
    return search


def test_search_is_still_a_factor_optimiser():
    assert isinstance(af.m.MockSearch(), AbstractFactorOptimiser)
    assert issubclass(af.Emcee, AbstractFactorOptimiser)
    assert issubclass(SearchFactorOptimiser, AbstractFactorOptimiser)


def test_two_ep_steps_run_one_search_per_factor_per_step(search):
    """
    The search is the graph's default optimiser, so it fits the two analysis factors
    and the two prior factors, once per EP step each.
    """
    factor_names = set(search.optimisation_counter)

    assert len(factor_names) == 4
    assert all(count == MAX_STEPS for count in search.optimisation_counter.values())
    assert sorted(step["analysis_name"] for step in search.steps) == sorted(
        f"{name}/optimization_{number}"
        for name in factor_names
        for number in range(MAX_STEPS)
    )


def test_each_step_gets_fresh_paths_and_a_rebuilt_updater(search):
    paths = [step["paths"] for step in search.steps]
    updaters = [step["updater"] for step in search.steps]

    assert len({id(p) for p in paths}) == len(paths)
    assert len({id(u) for u in updaters}) == len(updaters)
    assert all(step["updater_paths_current"] for step in search.steps)


def test_visual_switches_per_step_and_restored(search):
    for step in search.steps:
        assert step["visualize_fit"] is False
        assert step["visualize_before_fit"] is step["analysis_name"].endswith(
            "optimization_0"
        )

    assert search._visualize_fit is True
    assert search._visualize_before_fit is True


def test_search_internal_released_every_step(search):
    assert len(search.results) == 4 * MAX_STEPS
    assert all(result.released for result in search.results)


def test_optimise_delegates_to_the_adapter(factor_graph_model, monkeypatch):
    calls = []

    def recording_optimise(self, factor_approx, status):
        calls.append(self.search)
        return "mean_field", status

    monkeypatch.setattr(SearchFactorOptimiser, "optimise", recording_optimise)

    search = af.m.MockSearch()
    approx = factor_graph_model.mean_field_approximation().factor_approximation(
        factor_graph_model.graph.factors[0]
    )
    mean_field, _ = search.optimise(approx)

    assert mean_field == "mean_field"
    assert calls == [search]


def test_non_analysis_factor_is_rejected():
    class _Factor:
        pass

    class _Approx:
        factor = _Factor()

    with pytest.raises(NotImplementedError, match="can only be applied to"):
        SearchFactorOptimiser(af.m.MockSearch()).optimise(_Approx())
