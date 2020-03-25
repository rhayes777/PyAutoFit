import pytest

import autofit as af


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.UniformPrior()


class TestAddition:
    def test_prior_plus_prior(self, prior):
        sum_prior = prior + prior
        assert sum_prior.instance_from_unit_vector([1.0]) == 2.0

    def test_negative_prior(self, prior):
        negative = -prior
        assert negative.instance_from_unit_vector([1.0]) == -1.0

    def test_prior_minus_prior(self, prior):
        sum_prior = prior - prior
        assert sum_prior.instance_from_unit_vector([1.0]) == 0.0


class TestMultiplication:
    def test_prior_times_prior(self, prior):
        multiple_prior = (prior + prior) * (prior + prior)
        assert multiple_prior.instance_from_unit_vector([1.0]) == 4
