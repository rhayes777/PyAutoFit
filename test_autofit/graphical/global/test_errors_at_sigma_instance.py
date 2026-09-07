import pytest

import autofit as af
import autofit.graphical as g


class Analysis(af.Analysis):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def log_likelihood_function(self, instance):
        return -((instance.one - self.value) ** 2)


@pytest.fixture(name="model")
def make_model():
    """
    A global model containing a `Model(af.GaussianPrior)` (contributed by the
    hierarchical factor's distribution model) alongside an ordinary class component.
    """
    hierarchical_factor = g.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(mean=0.5, sigma=0.1),
        sigma=af.GaussianPrior(mean=1.0, sigma=0.01),
    )

    model_factor = g.AnalysisFactor(
        af.Collection(one=af.UniformPrior()),
        Analysis(0.5),
    )
    ordinary_factor = g.AnalysisFactor(
        af.Model(af.m.MockClassx2),
        Analysis(0.0),
    )

    hierarchical_factor.add_drawn_variable(model_factor.one)

    return g.FactorGraphModel(
        hierarchical_factor,
        model_factor,
        ordinary_factor,
    ).global_prior_model


@pytest.fixture(name="samples")
def make_samples(model):
    parameters = [
        [(j + 1) * (0.1 + 0.02 * i) for j in range(model.prior_count)]
        for i in range(10)
    ]

    return af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=list(range(10)),
            log_prior_list=10 * [0.0],
            weight_list=10 * [0.1],
        ),
    )


def _index_of(model, prior):
    return [
        prior_tuple.prior.id for prior_tuple in model.prior_tuples_ordered_by_id
    ].index(prior.id)


def _distribution_model(model):
    for component in model:
        if hasattr(component, "distribution_model"):
            return component.distribution_model
    raise AssertionError("no hierarchical distribution model in the global model")


def _prior_valued_instance(instance):
    for component in instance:
        if hasattr(component, "distribution_model"):
            return component.distribution_model
    raise AssertionError("no hierarchical distribution model in the instance")


def _ordinary_instance(instance):
    for component in instance:
        if hasattr(component, "two"):
            return component
    raise AssertionError("no ordinary class component in the instance")


@pytest.mark.parametrize("method", ["errors_at_sigma", "values_at_sigma"])
def test__tuple_valued_instance_for_prior_valued_component(model, samples, method):
    """
    `errors_at_sigma` / `values_at_sigma` are tuple valued. A `Model(af.GaussianPrior)`
    component cannot be constructed from those tuples, so the instance stores them as
    attributes instead of raising a `broadcast_arrays` `TypeError`.
    """
    values = getattr(samples, method)(sigma=1.0, as_instance=False)
    instance = getattr(samples, method)(sigma=1.0)

    distribution_model = _distribution_model(model)

    assert _prior_valued_instance(instance).mean == values[
        _index_of(model, distribution_model.mean)
    ]
    assert _prior_valued_instance(instance).sigma == values[
        _index_of(model, distribution_model.sigma)
    ]

    ordinary = _ordinary_instance(instance)

    assert ordinary.one in values
    assert ordinary.two in values


def test__median_pdf_instance_still_builds_a_real_prior(model, samples):
    """
    The scalar path is untouched: a `Model(af.GaussianPrior)` still constructs a prior.
    """
    distribution_model = _prior_valued_instance(samples.median_pdf())

    assert isinstance(distribution_model, af.GaussianPrior)
    assert isinstance(distribution_model.mean, float)
    assert isinstance(distribution_model.sigma, float)
