import autofit as af
import pytest


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.UniformPrior()


class TestAddition:
    def test_prior_plus_prior(self, prior):
        sum_prior = prior + prior
        assert sum_prior
