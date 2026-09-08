"""
Model assertions on the JAX paths.

An assertion signals failure by raising `FitException`, and a `raise` cannot happen inside a
trace: `check_assertions` applies a Python `not` to the assertion's value, and `CompoundAssertion`
combined its halves with a Python `and`, both of which coerce a tracer to a bool and raise
`TracerBoolConversionError`. So a JAX analysis either let the exception escape into the search
(non-vmapped) or failed to trace at all (`use_jax_vmap=True`, which is the Nautilus default for a
JAX analysis).

`Fitness` now builds the instance with `ignore_assertions=True` and applies the assertions as a
traced boolean through `xp.where`, mapping a violating model to `resample_figure_of_merit` — the
same value the numpy path reaches by catching the exception. These tests pin that equivalence on
all three JAX call surfaces (`call`, `_jit`, `_vmap`), the last being the strict one: under `vmap`
every parameter is a tracer, so an assertion that had kept any Python branch would fail there even
where jit-on-concrete passed.

JAX is an optional dependency and unit tests are the always-green numpy layer, so the file skips
whole rather than importing `jax` unconditionally (the numpy half is covered in
`test_fitness_assertions.py`).
"""

import numpy as np
import pytest

import autofit as af
from autofit.non_linear import fitness as fitness_module
from autofit.non_linear.fitness import Fitness

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


#: `gaussian_0.centre` is index 0 and `gaussian_1.centre` is index 3: the vector is ordered by
#: prior id, which is centre, normalization, sigma for each Gaussian in turn.
SATISFYING = [20.0, 1.0, 1.0, 10.0, 1.0, 1.0]
VIOLATING = [10.0, 1.0, 1.0, 20.0, 1.0, 1.0]

#: The finite sentinel Nautilus and dynesty pass, rather than `-inf`, so a rejection is
#: distinguishable from a merely terrible likelihood in the assertions below.
RESAMPLE = -1.0e99


def _model(assertion="ordering"):
    """
    Two Gaussians in a `Collection`. `child` attaches the assertion to `gaussian_0` rather than to
    the `Collection` — the form the pre-existing numpy regression test uses, and the one the
    traced path has to gather from the tree rather than read off the model it is handed.
    """
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)

    if assertion == "child":
        gaussian_0.add_assertion(gaussian_0.centre > gaussian_1.centre)

    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)

    if assertion == "ordering":
        model.add_assertion(model.gaussian_0.centre > model.gaussian_1.centre)
    elif assertion == "compound_prior":
        model.add_assertion(
            (model.gaussian_0.centre**2 + model.gaussian_0.normalization**2)
            > (model.gaussian_1.centre**2 + model.gaussian_1.normalization**2)
        )

    return model


def _fitness(model, use_jax, **kwargs):
    data = np.ones(20)
    noise_map = np.ones(20) * 0.1
    return Fitness(
        model=model,
        analysis=af.ex.Analysis(data=data, noise_map=noise_map, use_jax=use_jax),
        resample_figure_of_merit=RESAMPLE,
        **kwargs,
    )


def _numpy_figure_of_merit(model, vector):
    """The value the numpy path returns for the same model and vector — the reference."""
    return _fitness(model, use_jax=False).call(vector)


def test_jax_call_rejects_a_violating_vector():
    """
    The non-vmapped JAX path. Before the traced penalty this raised `FitException` out of
    `instance_from_vector`, since only the numpy branch of `call` has a `try/except`.
    """
    model = _model()
    fitness = _fitness(model, use_jax=True)

    assert fitness._is_jax is True
    assert fitness._apply_assertions_traced is True
    assert float(fitness.call(jnp.array(VIOLATING))) == RESAMPLE


def test_jax_call_passes_a_satisfying_vector():
    model = _model()
    fitness = _fitness(model, use_jax=True)

    figure_of_merit = float(fitness.call(jnp.array(SATISFYING)))

    assert figure_of_merit != RESAMPLE
    assert figure_of_merit == pytest.approx(
        _numpy_figure_of_merit(model, SATISFYING), abs=1.0e-8
    )


def test_jit_path_rejects_a_violating_vector_and_passes_a_satisfying_one():
    """
    Under `jit` the assertion must trace: whether the penalty is applied at all is a static
    Python bool set in `__init__`, and the verdict itself is a traced boolean fed to `xp.where`.
    """
    model = _model()
    fitness = _fitness(model, use_jax=True, use_jax_jit=True)

    assert fitness._call is fitness._jit
    assert float(fitness._call(jnp.array(VIOLATING))) == RESAMPLE
    assert float(fitness._call(jnp.array(SATISFYING))) == pytest.approx(
        _numpy_figure_of_merit(model, SATISFYING), abs=1.0e-8
    )


def test_vmap_path_rejects_only_the_violating_member_of_a_batch():
    """
    The strict check, and the configuration Nautilus actually builds for a JAX analysis. Both
    vectors are traced together, so the rejection has to be a value selected per batch member
    rather than a branch taken once.
    """
    model = _model()
    fitness = _fitness(model, use_jax=True, use_jax_vmap=True)

    assert fitness._call is fitness._vmap

    figures_of_merit = np.asarray(
        fitness._call(jnp.array([VIOLATING, SATISFYING]))
    )

    assert figures_of_merit.shape == (2,)
    assert figures_of_merit[0] == RESAMPLE
    assert figures_of_merit[1] == pytest.approx(
        _numpy_figure_of_merit(model, SATISFYING), abs=1.0e-8
    )


@pytest.mark.parametrize("use_jax_jit, use_jax_vmap", [(False, False), (True, False), (False, True)])
def test_compound_prior_assertion_on_every_jax_surface(use_jax_jit, use_jax_vmap):
    """
    The motivating case is an ordering between two *derived* quantities — the squared radius of a
    pair of shared priors, the 1D stand-in for the MGE `ell_comps` label degeneracy. That routes
    through `SumPrior` and `PowerPrior`, which must realise their operands with the same `xp` as
    the rest of the trace.
    """
    model = _model(assertion="compound_prior")
    fitness = _fitness(
        model, use_jax=True, use_jax_jit=use_jax_jit, use_jax_vmap=use_jax_vmap
    )

    # |(0.4, 0.3)|^2 = 0.25 > |(0.05, 0.05)|^2 = 0.005
    satisfying = [0.4, 0.3, 1.0, 0.05, 0.05, 1.0]
    violating = [0.05, 0.05, 1.0, 0.4, 0.3, 1.0]

    if use_jax_vmap:
        values = np.asarray(fitness._call(jnp.array([violating, satisfying])))
    else:
        values = np.array(
            [
                float(fitness._call(jnp.array(violating))),
                float(fitness._call(jnp.array(satisfying))),
            ]
        )

    assert values[0] == RESAMPLE
    assert values[1] == pytest.approx(
        _numpy_figure_of_merit(model, satisfying), abs=1.0e-8
    )


@pytest.mark.parametrize("use_jax_jit, use_jax_vmap", [(False, False), (True, False), (False, True)])
def test_model_without_assertions_is_unaffected(use_jax_jit, use_jax_vmap):
    """
    With nothing attached the `where` is not emitted at all, and the figure of merit must be the
    likelihood untouched — the penalty must not cost the common unconstrained model anything.
    """
    model = _model(assertion=None)
    fitness = _fitness(
        model, use_jax=True, use_jax_jit=use_jax_jit, use_jax_vmap=use_jax_vmap
    )

    assert fitness._apply_assertions_traced is False

    if use_jax_vmap:
        values = np.asarray(fitness._call(jnp.array([VIOLATING, SATISFYING])))
    else:
        values = np.array(
            [
                float(fitness._call(jnp.array(VIOLATING))),
                float(fitness._call(jnp.array(SATISFYING))),
            ]
        )

    assert np.all(values != RESAMPLE)
    assert values[0] == pytest.approx(
        _numpy_figure_of_merit(model, VIOLATING), abs=1.0e-8
    )
    assert values[1] == pytest.approx(
        _numpy_figure_of_merit(model, SATISFYING), abs=1.0e-8
    )


@pytest.mark.parametrize("use_jax_jit, use_jax_vmap", [(False, False), (True, False), (False, True)])
def test_child_attached_assertion_is_enforced_on_every_jax_surface(
    use_jax_jit, use_jax_vmap
):
    """
    The trap this closes: an assertion attached to a child `Model` is enforced on numpy, because
    that child checks its own assertions as its instance is built, but the traced path is handed
    only the top-level `Collection`. Reading that model's own `_assertions` would find nothing and
    silently sample violating models under JAX while rejecting them on numpy.
    """
    model = _model(assertion="child")
    fitness = _fitness(
        model, use_jax=True, use_jax_jit=use_jax_jit, use_jax_vmap=use_jax_vmap
    )

    assert model._assertions == []
    assert fitness._apply_assertions_traced is True

    if use_jax_vmap:
        values = np.asarray(fitness._call(jnp.array([VIOLATING, SATISFYING])))
    else:
        values = np.array(
            [
                float(fitness._call(jnp.array(VIOLATING))),
                float(fitness._call(jnp.array(SATISFYING))),
            ]
        )

    assert values[0] == RESAMPLE
    assert values[1] == pytest.approx(
        _numpy_figure_of_merit(model, SATISFYING), abs=1.0e-8
    )


# ---------------------------------------------------------------------------
# The figure-of-merit conversions must not touch the sentinel.
# ---------------------------------------------------------------------------

#: The three JAX call surfaces, as (use_jax_jit, use_jax_vmap) pairs.
SURFACES = [(False, False), (True, False), (False, True)]


def _values(fitness, vectors):
    """Evaluate `vectors` on whichever surface `fitness` dispatches to, as a numpy array."""
    if fitness.use_jax_vmap:
        return np.asarray(fitness._call(jnp.array(vectors)))
    return np.array([float(fitness._call(jnp.array(v))) for v in vectors])


@pytest.mark.parametrize("use_jax_jit, use_jax_vmap", SURFACES)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"convert_to_chi_squared": True},
        {"fom_is_log_likelihood": False},
        {"convert_to_chi_squared": True, "fom_is_log_likelihood": False},
    ],
)
def test_the_sentinel_survives_the_figure_of_merit_conversions(
    use_jax_jit, use_jax_vmap, kwargs
):
    """
    numpy returns `resample_figure_of_merit` from an early `return`, *before* the log prior is
    added and before the chi-squared multiply. A `where` applied to the log likelihood instead of
    to the final figure of merit would let both conversions rewrite the sentinel -- and
    `convert_to_chi_squared` flips its sign, so a rejected model would come back as `+2e99`: the
    single most attractive point in the space for a minimizer. The two backends must return the
    same number.
    """
    model = _model()
    fitness = _fitness(
        model, use_jax=True, use_jax_jit=use_jax_jit, use_jax_vmap=use_jax_vmap, **kwargs
    )
    numpy_fitness = _fitness(model, use_jax=False, **kwargs)

    values = _values(fitness, [VIOLATING, SATISFYING])

    assert values[0] == RESAMPLE
    assert values[0] == numpy_fitness.call(VIOLATING)
    assert values[1] == pytest.approx(numpy_fitness.call(SATISFYING), abs=1.0e-8)


def test_the_sentinel_survives_a_log_prior_that_is_not_zero():
    """
    The posterior-mode half of the check needs a model whose summed log prior is non-zero -- with
    the default `UniformPrior`s it is exactly `0.0`, so adding it to the sentinel would be
    invisible.
    """
    gaussian_0 = af.Model(af.ex.Gaussian)
    gaussian_1 = af.Model(af.ex.Gaussian)
    gaussian_0.centre = af.GaussianPrior(mean=0.0, sigma=30.0)
    gaussian_1.centre = af.GaussianPrior(mean=0.0, sigma=30.0)
    model = af.Collection(gaussian_0=gaussian_0, gaussian_1=gaussian_1)
    model.add_assertion(model.gaussian_0.centre > model.gaussian_1.centre)

    assert np.sum(model.log_prior_list_from_vector(vector=VIOLATING)) != 0.0

    fitness = _fitness(model, use_jax=True, fom_is_log_likelihood=False)

    assert float(fitness.call(jnp.array(VIOLATING))) == RESAMPLE


# ---------------------------------------------------------------------------
# State restored from a pickle.
# ---------------------------------------------------------------------------


def test_unpickling_without_the_assertion_state_still_enforces_assertions():
    """
    A `Fitness` pickled before the traced penalty existed carries neither key. Defaulting them to
    "no assertions" would leave a *resumed* JAX search quietly sampling models the user forbade,
    so they are recomputed from the restored model.
    """
    model = _model()
    fitness = _fitness(model, use_jax=True)

    state = fitness.__getstate__()
    del state["_traced_assertions"]
    del state["_apply_assertions_traced"]

    restored = Fitness.__new__(Fitness)
    restored.__setstate__(state)

    assert restored._apply_assertions_traced is True
    assert len(restored._traced_assertions) == 1
    assert float(restored.call(jnp.array(VIOLATING))) == RESAMPLE
    assert float(restored.call(jnp.array(SATISFYING))) == pytest.approx(
        _numpy_figure_of_merit(model, SATISFYING), abs=1.0e-8
    )


# ---------------------------------------------------------------------------
# Compound chaining, boundaries, singular operands and the config override.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_jax_jit, use_jax_vmap", SURFACES)
def test_compound_assertion_chain_on_every_jax_surface(use_jax_jit, use_jax_vmap):
    """
    `(a > b) > c` chains to two comparisons combined by `CompoundAssertion`, which is the class
    that had to give up its Python `and` to trace at all.
    """
    model = af.Collection(
        gaussian_0=af.Model(af.ex.Gaussian),
        gaussian_1=af.Model(af.ex.Gaussian),
        gaussian_2=af.Model(af.ex.Gaussian),
    )
    model.add_assertion(
        (model.gaussian_0.centre > model.gaussian_1.centre) > model.gaussian_2.centre
    )

    ordered = [3.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    second_half_violated = [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0]

    fitness = _fitness(
        model, use_jax=True, use_jax_jit=use_jax_jit, use_jax_vmap=use_jax_vmap
    )

    values = _values(fitness, [second_half_violated, ordered])

    assert values[0] == RESAMPLE
    assert values[1] == pytest.approx(
        _numpy_figure_of_merit(model, ordered), abs=1.0e-8
    )


@pytest.mark.parametrize("operator, expected_resample", [(">", True), (">=", False)])
def test_equality_boundary(operator, expected_resample):
    """
    The equality boundary is the only input that distinguishes `>` from `>=`, so it is what would
    catch the two being wired to the same comparison when `xp` was threaded through them.
    """
    model = af.Collection(
        gaussian_0=af.Model(af.ex.Gaussian),
        gaussian_1=af.Model(af.ex.Gaussian),
    )
    if operator == ">":
        model.add_assertion(model.gaussian_0.centre > model.gaussian_1.centre)
    else:
        model.add_assertion(model.gaussian_0.centre >= model.gaussian_1.centre)

    equal = [10.0, 1.0, 1.0, 10.0, 1.0, 1.0]
    fitness = _fitness(model, use_jax=True)

    value = float(fitness.call(jnp.array(equal)))

    if expected_resample:
        assert value == RESAMPLE
    else:
        assert value == pytest.approx(
            _numpy_figure_of_merit(model, equal), abs=1.0e-8
        )


def test_a_singular_second_half_traces_rather_than_raising():
    """
    The numpy path short-circuits `(0 < p) < (1 / p)` so `1 / 0` is never evaluated. JAX cannot:
    `logical_and` evaluates both halves. That is safe there -- `1 / 0` is `inf`, not an exception —
    so the trace must simply complete, and a violating vector must still reach the sentinel.
    """
    prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    model = af.Collection(p=prior)
    model.add_assertion((0.5 < prior) < (1 / prior))

    fitness = _fitness(model, use_jax=True)

    # p = 0.25 fails the first half (0.5 < 0.25 is False).
    assert float(fitness.call(jnp.array([0.25]))) == RESAMPLE
    # p = 0.75 satisfies both (0.5 < 0.75 < 1.333).
    assert float(fitness.call(jnp.array([0.75]))) != RESAMPLE


def test_assertions_on_an_assertion_operand_are_enforced():
    """
    `derived = p + 1` carries its own assertion, which numpy fires when the operand is realised.
    The gather has to follow the assertion's operand graph to find it, or JAX accepts a vector
    numpy rejects.
    """
    prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    derived = prior + 1
    derived.add_assertion(prior > 0.5)

    model = af.Collection(p=prior)
    model.add_assertion(derived < 2)

    fitness = _fitness(model, use_jax=True)

    assert fitness._apply_assertions_traced is True
    assert float(fitness.call(jnp.array([0.25]))) == RESAMPLE
    assert float(fitness.call(jnp.array([0.75]))) != RESAMPLE


class _ExceptionOverrideConf:
    """
    Stands in for `autonerves.conf` with `general.test.exception_override` set, the switch that
    turns assertion checking off wholesale on the numpy path.
    """

    instance = {"general": {"test": {"exception_override": True}}}


def test_exception_override_disables_the_traced_penalty(monkeypatch):
    """
    `exception_override` exists so a test run can push a model through a search without its
    assertions rejecting anything. It disables `check_assertions` on numpy, so it has to disable
    the traced penalty too -- otherwise a JAX run under the override rejects models a numpy run
    accepts.
    """
    monkeypatch.setattr(fitness_module, "conf", _ExceptionOverrideConf())

    model = _model()
    fitness = _fitness(model, use_jax=True)

    assert fitness._apply_assertions_traced is False

    value = float(fitness.call(jnp.array(VIOLATING)))

    assert value != RESAMPLE

    # The reference is the same model without the assertion: the override is read from
    # `autofit.mapper.prior_model.abstract`'s own `conf` on the numpy path, which this
    # monkeypatch does not reach, so a numpy `Fitness` on the asserting model would still
    # resample. The likelihood depends only on the vector, so the assertion-free twin gives the
    # value the override is supposed to let through.
    assert value == pytest.approx(
        _numpy_figure_of_merit(_model(assertion=None), VIOLATING), abs=1.0e-8
    )
