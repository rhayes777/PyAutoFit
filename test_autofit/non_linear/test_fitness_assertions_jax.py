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
