import pytest

import autofit as af


@pytest.fixture(name="factor")
def make_factor(hierarchical_factor):
    return hierarchical_factor.factors[0]


def test_optimise(factor):
    optimizer = af.DynestyStatic(maxcall=100, dynamic_delta=False, delta=0.1,)

    _, status = optimizer.optimise(
        factor.mean_field_approximation().factor_approximation(factor)
    )
    assert status


def test_instance(factor):
    prior_model = factor.prior_model
    assert len(prior_model.priors) == 3
    assert factor.log_likelihood_function(
        prior_model.instance_from_prior_medians()
    ) == pytest.approx(-3.2215236261987186, 1e-5)
