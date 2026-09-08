import logging
import pickle

import numpy as np
import pytest

import autofit as af
from autofit.non_linear import fitness as fitness_module
from autofit.non_linear.fitness import Fitness
from test_autofit.non_linear.constant_analysis import ConstantAnalysis


def test_fitness_returns_resample_fom_on_assertion_failure():
    """
    If an added assertion rejects a sampled parameter vector, ``Fitness.call``
    must return ``resample_figure_of_merit`` rather than letting the
    ``FitException`` escape into the non-linear search. Regression for
    the DynestyStatic-blocking bug where ``instance_from_vector`` was
    called outside the ``try/except FitException`` block.
    """
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)
    gaussian_0.add_assertion(gaussian_0.centre > gaussian_1.centre)
    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    fitness = Fitness(model=model, analysis=analysis)

    # centre_0 (index 0) = 10 < centre_1 (index 3) = 20  → assertion violated
    violating = [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]
    assert fitness.call(violating) == fitness.resample_figure_of_merit

    # centre_0 = 20 > centre_1 = 10  → assertion satisfied, real FOM returned
    satisfying = [20.0, 1.0, 1.0, 10.0, 1.0, 1.0]
    assert fitness.call(satisfying) != fitness.resample_figure_of_merit


def test_numpy_path_is_unchanged_by_the_traced_assertion_penalty():
    """
    The traced penalty is JAX-only: on numpy the `FitException` raised by `instance_from_vector`
    is still what produces `resample_figure_of_merit`, and the `xp.where` added for JAX must not
    run here at all. `_apply_assertions_traced` is nonetheless `True` — it is a property of the
    *model*, decided once in `__init__` so it is a compile-time constant under `jax.jit` — so
    this pins that the flag being set does not perturb a numpy fit.
    """
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)
    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)
    model.add_assertion(model.gaussian_0.centre > model.gaussian_1.centre)

    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

    fitness = Fitness(model=model, analysis=analysis)

    assert fitness._is_jax is False
    assert fitness._apply_assertions_traced is True

    violating = [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]
    assert fitness.call(violating) == fitness.resample_figure_of_merit

    satisfying = [20.0, 1.0, 1.0, 10.0, 1.0, 1.0]
    instance = model.instance_from_vector(satisfying)
    assert fitness.call(satisfying) == pytest.approx(
        analysis.log_likelihood_function(instance=instance)
    )


def test_apply_assertions_traced_is_false_without_assertions():
    """
    With nothing attached the `where` would be a no-op, so it is not emitted at all — one fewer
    traced operation in every likelihood evaluation of the (overwhelmingly common) unconstrained
    model.
    """
    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    fitness = Fitness(
        model=af.Model(af.ex.Gaussian),
        analysis=af.ex.Analysis(data=data, noise_map=noise_map),
    )

    assert fitness._apply_assertions_traced is False


def test_gathered_assertions_survive_a_pickle_roundtrip():
    """
    A search pickles its `Fitness` (parallel workers, resume). The gathered assertion list is
    plain model objects, so it pickles -- and the priors inside it are the same objects as those
    in the pickled model, since pickle preserves identity within one blob, so the restored
    assertions still key into the restored model's arguments.
    """
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)
    gaussian_0.add_assertion(gaussian_0.centre > gaussian_1.centre)
    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    fitness = Fitness(
        model=model,
        analysis=af.ex.Analysis(data=data, noise_map=noise_map),
        resample_figure_of_merit=-1.0e99,
    )

    restored = pickle.loads(pickle.dumps(fitness))

    assert restored._apply_assertions_traced is True
    assert len(restored._traced_assertions) == 1
    assert restored.call([10.0, 1.0, 1.0, 20.0, 1.0, 1.0]) == -1.0e99
    assert (
        bool(
            restored.model.assertions_satisfied_from_vector(
                [10.0, 1.0, 1.0, 20.0, 1.0, 1.0],
                assertions=restored._traced_assertions,
            )
        )
        is False
    )


def test_unpickling_an_old_fitness_recomputes_the_gathered_assertions():
    """
    A `Fitness` pickled before the traced penalty existed carries neither `_traced_assertions` nor
    `_apply_assertions_traced`. Defaulting them to "no assertions" would mean a search *resumed*
    from such a pickle silently stops enforcing its model's assertions under JAX -- the failure is
    invisible, since the fit runs happily and simply samples models the user forbade. The state is
    recomputed from the restored model instead, which is where the assertions live anyway.
    """
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)
    gaussian_0.add_assertion(gaussian_0.centre > gaussian_1.centre)
    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    fitness = Fitness(
        model=model,
        analysis=af.ex.Analysis(data=data, noise_map=noise_map),
        resample_figure_of_merit=-1.0e99,
    )

    state = fitness.__getstate__()
    del state["_traced_assertions"]
    del state["_apply_assertions_traced"]

    restored = Fitness.__new__(Fitness)
    restored.__setstate__(state)

    assert restored._apply_assertions_traced is True
    assert len(restored._traced_assertions) == 1
    assert restored.call([10.0, 1.0, 1.0, 20.0, 1.0, 1.0]) == -1.0e99


def test_numpy_short_circuit_assertion_returns_the_sentinel():
    """
    The numpy `and` short-circuit is part of the user contract: a chained assertion whose second
    half is singular exactly where the first fails must reach the search as a resample, not as a
    `ZeroDivisionError` out of the likelihood.
    """
    prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    model = af.Collection(p=prior)
    model.add_assertion((0 < prior) < (1 / prior))

    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    fitness = Fitness(
        model=model,
        analysis=af.ex.Analysis(data=data, noise_map=noise_map),
        resample_figure_of_merit=-1.0e99,
    )

    assert fitness.call([0.0]) == -1.0e99


@pytest.mark.parametrize(
    "kwargs",
    [
        {"convert_to_chi_squared": True},
        {"fom_is_log_likelihood": False},
        {"convert_to_chi_squared": True, "fom_is_log_likelihood": False},
    ],
)
def test_numpy_returns_the_raw_sentinel_whatever_the_conversion(kwargs):
    """
    The reference the JAX path must match: numpy returns `resample_figure_of_merit` from an early
    `return`, so neither the log-prior addition nor the chi-squared multiply touches it.
    """
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)
    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)
    model.add_assertion(model.gaussian_0.centre > model.gaussian_1.centre)

    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    fitness = Fitness(
        model=model,
        analysis=af.ex.Analysis(data=data, noise_map=noise_map),
        resample_figure_of_merit=-1.0e99,
        **kwargs,
    )

    assert fitness.call([10.0, 1.0, 1.0, 20.0, 1.0, 1.0]) == -1.0e99


def _ceiling_fitness(log_likelihood, ceiling=None, **kwargs):
    """
    A `Fitness` whose analysis returns `log_likelihood` for any instance.

    `ceiling` is set on the built object rather than pushed through the config, because the
    packaged default is now *disabled* — every test of the enabled path has to opt in, exactly as a
    user's `general.yaml` would.
    """
    fitness = Fitness(
        model=af.Model(af.ex.Gaussian),
        analysis=ConstantAnalysis(log_likelihood=log_likelihood),
        **kwargs,
    )

    if ceiling is not None:
        fitness.log_likelihood_ceiling = ceiling

    return fitness


PARAMETERS = [1.0, 1.0, 1.0]

#: The ceiling PyAutoFit used to ship enabled, and the one `autolens_profiling` opts into. Every
#: enabled-path test below sets it explicitly.
CEILING = 1.0e20


@pytest.fixture(autouse=True)
def _reset_ceiling_warning(monkeypatch):
    """
    The first-fire warning is a module-level latch, so a test that fires the guard would otherwise
    silence every test that runs after it. `monkeypatch` restores the flag on teardown.
    """
    monkeypatch.setattr(fitness_module, "_log_likelihood_ceiling_warning_emitted", False)


def test_fitness_ceiling_defaults_to_disabled():
    """
    The packaged `general.yaml` ships the ceiling blank, so the guard is **off** unless a config
    opts in. The threshold is a bare magnitude, and a log likelihood scales with the noise-map
    units (`chi_squared` as `noise ** -2`, `noise_normalization` linearly in the pixel count), so a
    fixed value can reject a legitimate fit on a badly-scaled dataset.

    It is still read once, in `__init__`, and is still a static Python float — `Fitness.call`
    compares it against a traced value under `jax.jit` / `jax.vmap`, which a config lookup per call
    could not do.
    """
    fitness = _ceiling_fitness(log_likelihood=1.0)

    assert fitness.log_likelihood_ceiling == float("inf")
    assert type(fitness.log_likelihood_ceiling) is float


def test_fitness_with_the_default_ceiling_passes_an_enormous_finite_value():
    """
    The consequence of the default being off: a `1e266` log likelihood is returned untouched. That
    is the deliberate trade — the fp64-Cholesky garbage this guard was built for is let through
    unless a config opts in, so that a legitimately huge likelihood is never silently rejected.
    """
    fitness = _ceiling_fitness(log_likelihood=1.0e266)

    assert fitness.call(PARAMETERS) == 1.0e266


@pytest.mark.parametrize("log_likelihood", [1.0e266, -1.0e266, 1.0e21])
def test_fitness_rejects_log_likelihoods_above_the_ceiling(log_likelihood):
    """
    With the guard enabled, a finite log likelihood above the ceiling is numerical garbage — an
    fp64 Cholesky on a non-positive-definite matrix returns values up to `3e+303` — and must be
    mapped to `resample_figure_of_merit` rather than accepted as the search's best point. Both
    signs are rejected, since the guard is on the magnitude.
    """
    fitness = _ceiling_fitness(log_likelihood=log_likelihood, ceiling=CEILING)

    assert fitness.call(PARAMETERS) == fitness.resample_figure_of_merit


@pytest.mark.parametrize("log_likelihood", [1.0e19, -1.0e19, 30701.3, 0.0])
def test_fitness_passes_log_likelihoods_below_the_ceiling(log_likelihood):
    """
    Anything of a plausible magnitude is returned untouched — the guard must not perturb ordinary
    likelihoods, including the ~3e4 values a real lens fit reaches.
    """
    fitness = _ceiling_fitness(log_likelihood=log_likelihood, ceiling=CEILING)

    assert fitness.call(PARAMETERS) == log_likelihood


def test_fitness_ceiling_disabled_passes_any_finite_value():
    """
    A `null` ceiling in the config disables the guard, which `get_log_likelihood_ceiling` expresses
    as `inf` — `abs(x) > inf` is never true, so the `where` becomes a no-op rather than a branch.
    """
    fitness = _ceiling_fitness(log_likelihood=1.0e266, ceiling=float("inf"))

    assert fitness.call(PARAMETERS) == 1.0e266


def test_fitness_ceiling_leaves_the_nan_and_inf_guards_intact():
    """
    The magnitude guard sits after the isnan/isinf `where`s, so the sentinel those produce must
    survive it rather than being re-flagged into something else. The NaN/inf guards are
    unconditional — unlike the ceiling, they are not config-gated.
    """
    assert _ceiling_fitness(log_likelihood=np.nan, ceiling=CEILING).call(
        PARAMETERS
    ) == _ceiling_fitness(log_likelihood=1.0).resample_figure_of_merit
    assert _ceiling_fitness(log_likelihood=np.inf, ceiling=CEILING).call(
        PARAMETERS
    ) == _ceiling_fitness(log_likelihood=1.0).resample_figure_of_merit


def test_fitness_ceiling_is_idempotent_against_a_finite_resample_sentinel():
    """
    Searches that cannot use `-inf` pass a large negative sentinel (dynesty/nautilus use `-1e99`)
    as `resample_figure_of_merit`. Its own magnitude is above the ceiling, so the guard must map it
    to itself rather than oscillating.
    """
    fitness = _ceiling_fitness(
        log_likelihood=1.0e266, ceiling=CEILING, resample_figure_of_merit=-1.0e99
    )

    assert fitness.call(PARAMETERS) == -1.0e99


def test_fitness_warns_the_first_time_the_ceiling_fires(caplog):
    """
    A rejection is invisible from inside the search — it looks exactly like a bad model — so the
    numpy path says so once. The message names the value, the ceiling, and the units caveat, since
    only the user can tell garbage from a legitimately enormous likelihood.
    """
    fitness = _ceiling_fitness(log_likelihood=1.0e266, ceiling=CEILING)

    with caplog.at_level(logging.WARNING, logger=fitness_module.__name__):
        fitness.call(PARAMETERS)

    assert len(caplog.records) == 1
    assert "1e+266" in caplog.text
    assert "noise-map units" in caplog.text


def test_fitness_warns_only_once_per_process(caplog):
    """
    A nested sampler can reject millions of samples in a run; one warning is information, one per
    sample is a wall of text that hides everything else. The latch is module-level, so a fitness
    object rebuilt on resume does not re-warn either.
    """
    fitness = _ceiling_fitness(log_likelihood=1.0e266, ceiling=CEILING)

    with caplog.at_level(logging.WARNING, logger=fitness_module.__name__):
        fitness.call(PARAMETERS)
        fitness.call(PARAMETERS)
        _ceiling_fitness(log_likelihood=1.0e266, ceiling=CEILING).call(PARAMETERS)

    assert len(caplog.records) == 1


def test_fitness_does_not_warn_when_nothing_is_rejected(caplog):
    """
    The warning is tied to an actual rejection, not to the guard being configured — an enabled
    ceiling that never fires must stay silent.
    """
    fitness = _ceiling_fitness(log_likelihood=30701.3, ceiling=CEILING)

    with caplog.at_level(logging.WARNING, logger=fitness_module.__name__):
        fitness.call(PARAMETERS)

    assert caplog.records == []


class _StubConf:
    """
    Stands in for `autonerves.conf`, so the config-reading half of the guard can be tested without
    pushing a config directory.
    """

    def __init__(self, value=None, present=True):
        test = {"log_likelihood_ceiling": value} if present else {}
        self.instance = {"general": {"test": test}}


@pytest.mark.parametrize(
    "value, expected",
    [
        # PyYAML parses a bare `1e20` as a *string* (it requires a decimal point or a signed
        # exponent to resolve a float), so the reader must coerce rather than trust.
        ("1e20", 1.0e20),
        (1.0e20, 1.0e20),
        (30701.0, 30701.0),
        # Disabled, in each of the ways a config can express it.
        (None, float("inf")),
        (float("inf"), float("inf")),
        ("not-a-number", float("inf")),
        # A non-positive ceiling would reject every model; disable rather than wedge the fit.
        (0.0, float("inf")),
        (-1.0, float("inf")),
    ],
)
def test_get_log_likelihood_ceiling_reads_the_config(monkeypatch, value, expected):
    monkeypatch.setattr(fitness_module, "conf", _StubConf(value=value))

    assert fitness_module.get_log_likelihood_ceiling() == expected


def test_get_log_likelihood_ceiling_is_disabled_when_the_key_is_absent(monkeypatch):
    """
    A workspace whose `general.yaml` pre-dates the key must inherit **today's** default, which is
    disabled — not the `1e20` that shipped with the guard's first release. A missing key is silence
    about the guard, and silence means off.
    """
    monkeypatch.setattr(fitness_module, "conf", _StubConf(present=False))

    assert fitness_module.LOG_LIKELIHOOD_CEILING_DEFAULT == float("inf")
    assert (
        fitness_module.get_log_likelihood_ceiling()
        == fitness_module.LOG_LIKELIHOOD_CEILING_DEFAULT
    )
