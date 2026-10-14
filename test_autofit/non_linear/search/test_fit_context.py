"""
The minimal `run(ctx)` bridge (search-extensibility phase A2, `docs/design/run_ctx.md`).

`Drawer` and `Nautilus` run through `run(ctx)`; a search that overrides `_fit` keeps
working; a toy search written only against the frozen contract fits end to end.
"""
import numpy as np
import pytest

import autofit as af
from autofit.non_linear.search import capabilities as cap
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.search.fit_context import FitContext, RawSamples

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

FROZEN_MEMBERS = (
    "model",
    "paths",
    "objective",
    "fitness",
    "test_mode_level",
    "pool",
    "start_points",
    "rng",
    "resume",
    "checkpointer",
    "schedule",
    "update",
)


def _analysis():
    return af.ex.Analysis(
        data=np.ones(20), noise_map=np.ones(20) * 0.1, use_jax=False
    )


@pytest.mark.parametrize(
    "search_cls",
    [af.Drawer, af.Nautilus, af.Emcee, af.Zeus, af.BFGS, af.LBFGS],
)
def test_proofs_are_on_the_bridge(search_cls):
    assert search_cls.run is not NonLinearSearch.run
    assert search_cls.raw_samples_from is not NonLinearSearch.raw_samples_from
    assert search_cls._fit is NonLinearSearch._fit


@pytest.mark.parametrize(
    "search_cls, test_mode_samples",
    [(af.Emcee, True), (af.Zeus, False), (af.LBFGS, True)],
)
def test_ctx_start_points_forwards_the_search_test_mode_samples_choice(
    monkeypatch, search_cls, test_mode_samples
):
    """
    ``ctx.start_points`` lets the initializer return its test-mode samples unless the
    search opts out (``_start_points_in_test_mode``): Zeus always drew real start
    points (``test_mode_samples=False`` in its former ``_fit``).
    """
    from autofit.non_linear.search.fit_context import FitContext

    seen = {}

    def samples_from_model(total_points, model, fitness, paths, **kwargs):
        seen.update(kwargs)
        points = [[50.0, 1.0, 5.0]] * total_points
        return points, points, [0.0] * total_points

    search = search_cls()
    monkeypatch.setattr(search.initializer, "samples_from_model", samples_from_model)

    model = af.Model(af.ex.Gaussian)
    analysis = _analysis()
    fitness = search.make_fitness(analysis=analysis, model=model)

    ctx = FitContext(
        search=search,
        model=model,
        analysis=analysis,
        fitness=fitness,
        pool=search._pools(analysis),
        test_mode_level=1,
    )
    monkeypatch.setattr(search, "plot_start_point", lambda **kwargs: None)

    parameters, _ = ctx.start_points(3)

    assert parameters.shape == (3, 3)
    assert seen["test_mode_samples"] is test_mode_samples


def test_drawer_fit_runs_through_run_ctx(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")

    seen = []
    original = af.Drawer.run

    def spy(self, ctx):
        seen.append(ctx)
        return original(self, ctx)

    monkeypatch.setattr(af.Drawer, "run", spy)

    search = af.Drawer(total_draws=5)
    result = search.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert len(seen) == 1
    ctx = seen[0]
    assert isinstance(ctx, FitContext)
    for member in FROZEN_MEMBERS:
        assert hasattr(ctx, member), member
    assert ctx.test_mode_level == 1
    # A3: the archive strategy is the search's; Drawer has no resume state.
    assert ctx.checkpointer == search.checkpointer
    assert ctx.resume is None

    # Never stored on the search.
    assert not any(isinstance(value, FitContext) for value in vars(search).values())

    assert len(result.samples.sample_list) == 5


class ToySearch(NonLinearSearch):
    """
    A search written only against the frozen `run(ctx)` contract: draw points, evaluate
    them through the batched objective, keep them all.
    """

    posterior_kind = cap.PosteriorKind.POINT
    objective_target = cap.ObjectiveTarget(
        cap.ObjectiveQuantity.LOG_LIKELIHOOD, cap.CoordinateSpace.PHYSICAL
    )
    invalid_value = -float("inf")

    def run(self, ctx):
        parameters, _ = ctx.start_points(6)
        log_likelihoods = np.asarray(ctx.objective("batched")(parameters))
        return {"parameters": parameters, "log_likelihoods": log_likelihoods}

    def raw_samples_from(self, model, internal):
        return RawSamples(
            parameters=internal["parameters"].tolist(),
            log_likelihood=internal["log_likelihoods"].tolist(),
            info={"total_samples": len(internal["log_likelihoods"])},
        )

    def plot_results(self, samples):
        pass


def test_a_new_search_needs_only_run_and_raw_samples_from(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    search = ToySearch(initializer=af.InitializerPrior())
    model = af.Model(af.ex.Gaussian)
    result = search.fit(model=model, analysis=_analysis())

    assert len(result.samples.sample_list) == 6
    assert result.samples.max_log_likelihood() is not None


def test_failure_in_run_closes_the_context(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    closed = []

    def failing_run(self, ctx):
        ctx.objective("scalar")
        original_close = ctx.close

        def close(failed):
            closed.append(failed)
            return original_close(failed)

        ctx.close = close
        raise RuntimeError("backend failed")

    monkeypatch.setattr(ToySearch, "run", failing_run)

    with pytest.raises(RuntimeError, match="backend failed"):
        ToySearch().fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert closed == [True]


def test_a_search_with_neither_hook_is_abstract():
    import inspect

    class Empty(NonLinearSearch):
        pass

    assert inspect.isabstract(NonLinearSearch)
    assert inspect.isabstract(Empty)
    assert not inspect.isabstract(ToySearch)
    assert not inspect.isabstract(af.Drawer)

    with pytest.raises(TypeError):
        Empty()


def test_schedule_chunks_cover_the_budget():
    search = af.Drawer(iterations_per_full_update=4)
    from autofit.non_linear.search.fit_context import UpdateSchedule

    assert list(UpdateSchedule(search).chunks(10)) == [4, 4, 2]


def test_resume_is_the_resume_state_only_when_its_file_exists(tmp_path):
    from autofit.non_linear.checkpoint import NativeFileCheckpointer

    class ResumableToy(ToySearch):
        resume_state = NativeFileCheckpointer("toy_checkpoint.bin", loader=open)

    search = ResumableToy()
    search.paths = af.DirectoryPaths(name="toy", path_prefix=str(tmp_path))
    search.paths.model = af.Model(af.ex.Gaussian)

    def ctx():
        return FitContext(
            search=search,
            model=search.paths.model,
            analysis=None,
            fitness=None,
            pool=None,
            test_mode_level=0,
        )

    assert ctx().resume is None
    assert ctx().checkpointer is search.checkpointer

    path = search.resume_state.path(search.paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"state")

    assert ctx().resume is search.resume_state


# Review 05 findings 5 and 6: the frozen contract promises that the context closes the
# pools it built on every exit, and that `ctx.update` saves the checkpoint.


class _RecordingPool:
    def __init__(self, calls, processes, **kwargs):
        self.calls = calls
        calls.append("created")

    def close(self):
        self.calls.append("close")

    def terminate(self):
        self.calls.append("terminate")

    def join(self):
        self.calls.append("join")


@pytest.fixture(name="pool_calls")
def make_pool_calls(monkeypatch):
    from types import SimpleNamespace

    from autofit.non_linear.parallel import pool as pool_module

    calls = []
    monkeypatch.setattr(
        pool_module,
        "fork_context",
        lambda: SimpleNamespace(
            Pool=lambda processes, **kwargs: _RecordingPool(calls, processes, **kwargs)
        ),
    )
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    return calls


class PoolToySearch(ToySearch):
    fail = False

    def run(self, ctx):
        assert ctx.pool() is not None
        internal = super().run(ctx)
        if self.fail:
            raise RuntimeError("backend failed")
        return internal


def test_a_successful_run_closes_and_joins_the_pools_the_context_built(pool_calls):
    search = PoolToySearch(number_of_cores=2, initializer=af.InitializerPrior())
    search.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert pool_calls == ["created", "close", "join"]


def test_a_failed_run_terminates_and_joins_the_pools_the_context_built(pool_calls):
    search = PoolToySearch(number_of_cores=2, initializer=af.InitializerPrior())
    search.fail = True

    with pytest.raises(RuntimeError, match="backend failed"):
        search.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert pool_calls == ["created", "terminate", "join"]


def test_pool_factory_close_is_idempotent(pool_calls):
    from autofit.non_linear.parallel import PoolFactory

    factory = PoolFactory(ToySearch(number_of_cores=2))
    factory()
    factory.close(failed=False)
    factory.close(failed=True)

    assert pool_calls == ["created", "close", "join"]


def _recording_resume_state():
    from autofit.non_linear.checkpoint import DillCheckpointer

    class RecordingCheckpointer(DillCheckpointer):
        def __init__(self):
            super().__init__(filename="toy_resume.dill")
            self.saved = []
            self.finalized = []

        def save(self, paths, obj):
            self.saved.append(obj)
            super().save(paths, obj)

        def finalize(self, paths, obj=None):
            self.finalized.append(obj)

    return RecordingCheckpointer()


class UpdatingToySearch(ToySearch):
    interrupt = False

    def run(self, ctx):
        internal = super().run(ctx)
        ctx.update(internal)
        if self.interrupt:
            raise InterruptedError("killed after the update")
        return internal


def test_ctx_update_saves_the_resume_state_so_an_interrupted_run_keeps_it(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    strategy = _recording_resume_state()

    class Resumable(UpdatingToySearch):
        resume_state = strategy

    search = Resumable(
        initializer=af.InitializerPrior(), path_prefix=str(tmp_path), name="toy_update"
    )
    search.interrupt = True

    with pytest.raises(InterruptedError):
        search.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    assert len(strategy.saved) == 1
    assert len(strategy.saved[0]["log_likelihoods"]) == 6
    # Saved through `save`, never the completion `finalize`, and still on disk to resume.
    assert strategy.finalized == []
    assert strategy.exists(search.paths)


def test_ctx_update_saves_nothing_for_a_search_without_a_resume_state(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)

    saved = []
    search = UpdatingToySearch(
        initializer=af.InitializerPrior(), path_prefix=str(tmp_path), name="toy_plain"
    )
    monkeypatch.setattr(
        type(search.checkpointer), "save", lambda self, paths, obj: saved.append(obj)
    )
    search.fit(model=af.Model(af.ex.Gaussian), analysis=_analysis())

    # The archive is written once, at completion, not by the mid-run update.
    assert len(saved) <= 1
