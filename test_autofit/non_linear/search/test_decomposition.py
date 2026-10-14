"""
The decomposition of ``NonLinearSearch`` (search-extensibility A4a): the generic
test-mode reduction from ``test_mode_budget`` applied by the post-init hook, the
paths capabilities, ``paths_from`` and ``SearchRuntimeSettings``.
"""

from pathlib import Path

import pytest

import autofit as af
from autofit.non_linear.paths import paths_from
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.search import abstract_search
from autofit.non_linear.search.runtime import SearchRuntimeSettings

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="test_mode_1")
def make_test_mode_1(monkeypatch):
    monkeypatch.setenv("PYAUTO_TEST_MODE", "1")


@pytest.fixture(name="test_mode_off")
def make_test_mode_off(monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)


def test_abstract_search_is_at_most_600_lines():
    """The A4 witness: the decomposed module stays small."""
    lines = Path(abstract_search.__file__).read_text().splitlines()

    assert len(lines) <= 600


@pytest.mark.parametrize("cls", [af.BFGS, af.LBFGS])
def test_bfgs_runs_a_reduced_maxiter_at_test_mode_1(cls, test_mode_1):
    assert cls.test_mode_budget == {"maxiter": 10}
    assert cls().maxiter == 10
    assert cls(maxiter=500).maxiter == 10


@pytest.mark.parametrize("cls", [af.BFGS, af.LBFGS])
def test_bfgs_keeps_its_maxiter_outside_test_mode(cls, test_mode_off):
    assert cls().maxiter == 15000


@pytest.mark.parametrize(
    "cls, attributes",
    [
        (af.Emcee, {"nwalkers": 20, "nsteps": 10}),
        (af.Zeus, {"nwalkers": 20, "nsteps": 10}),
        (af.DynestyStatic, {"maxcall": 1}),
        (af.DynestyDynamic, {"maxcall": 1}),
        (af.Nautilus, {"n_like_max": 1}),
    ],
)
def test_generic_reduction_sets_the_budget(cls, attributes, test_mode_1):
    search = cls()

    for name, value in attributes.items():
        assert getattr(search, name) == value


def test_dotted_budget_keys_reach_members(test_mode_1):
    search = af.MultiStartAdam()

    assert search.convergence.window == 1
    assert search.convergence.min_steps == 1


def test_post_init_hook_runs_once_after_the_outermost_init(test_mode_1):
    calls = []

    class Subclass(af.Emcee):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Set after the base __init__: the reduction still wins, because the
            # post-init hook runs once the whole construction has returned.
            self.nsteps = 1000

        def apply_test_mode(self):
            calls.append(type(self).__name__)
            super().apply_test_mode()

    search = Subclass()

    assert calls == ["Subclass"]
    assert search.nsteps == 10


def test_no_reduction_outside_test_mode(test_mode_off):
    search = af.Emcee(nwalkers=60, nsteps=3000)

    assert (search.nwalkers, search.nsteps) == (60, 3000)


def test_empty_budget_is_a_no_op(test_mode_1):
    search = af.Drawer(total_draws=7)

    assert af.Drawer.test_mode_budget == {}
    assert search.total_draws == 7


def test_paths_capabilities(tmp_path):
    null = NullPaths()
    directory = af.DirectoryPaths(name="caps", path_prefix=str(tmp_path))

    assert (null.has_timer, null.supports_checkpointing) == (False, False)
    assert (directory.has_timer, directory.supports_checkpointing) == (True, True)
    assert af.DatabasePaths.has_timer is False
    assert af.DatabasePaths.supports_checkpointing is True


def test_paths_from():
    given = NullPaths()

    assert paths_from(paths=given) is given
    assert isinstance(paths_from(), NullPaths)
    assert isinstance(paths_from(name="name"), af.DirectoryPaths)
    assert isinstance(paths_from(path_prefix=Path("prefix")), af.DirectoryPaths)


def test_runtime_settings_are_applied_as_attributes():
    settings = SearchRuntimeSettings.from_config(iterations_per_quick_update=7)
    search = af.m.MockSearch(iterations_per_quick_update=7)

    for name in settings.__dataclass_fields__:
        assert getattr(search, name) == getattr(settings, name)
    assert search.iterations_per_quick_update == 7.0
