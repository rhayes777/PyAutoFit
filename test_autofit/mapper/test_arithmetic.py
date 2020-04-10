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

    def test_prior_plus_float(self, prior):
        sum_prior = prior + 1.0
        assert sum_prior.instance_from_unit_vector([1.0]) == 2.0

    def test_float_plus_prior(self, prior):
        sum_prior = 1.0 + prior
        assert sum_prior.instance_from_unit_vector([1.0]) == 2.0


class TestMultiplication:
    def test_prior_times_prior(self, prior):
        multiple_prior = (prior + prior) * (prior + prior)
        assert multiple_prior.instance_from_unit_vector([1.0]) == 4

    def test_prior_times_float(self, prior):
        multiple_prior = prior * 2.0
        assert multiple_prior.instance_from_unit_vector([1.0]) == 2.0

    def test_float_times_prior(self, prior):
        multiple_prior = 2.0 * prior
        assert multiple_prior.instance_from_unit_vector([1.0]) == 2.0


class TestDivision:
    def test_prior_over_prior(self, prior):
        division_prior = prior / prior
        assert division_prior.instance_from_unit_vector(
            [10.0],
            assert_priors_in_limits=False
        ) == 1

    def test_prior_over_float(self, prior):
        division_prior = prior / 2
        assert division_prior.instance_from_unit_vector(
            [1.0]
        ) == 0.5

    def test_float_over_prior(self, prior):
        division_prior = 4.0 / prior
        assert division_prior.instance_from_unit_vector(
            [0.5]
        ) == 8.0


class TestFloorDiv:
    def test_prior_over_int(self, prior):
        division_prior = prior // 2
        assert division_prior.instance_from_unit_vector(
            [3.0],
            assert_priors_in_limits=False
        ) == 1.0

    def test_int_over_prior(self, prior):
        division_prior = 3 // prior
        assert division_prior.instance_from_unit_vector(
            [2.0],
            assert_priors_in_limits=False
        ) == 1.0


class TestMod:
    def test_prior_mod_int(self, prior):
        mod_prior = prior % 3
        assert mod_prior.instance_from_unit_vector(
            [5.0],
            assert_priors_in_limits=False
        ) == 2.0

    def test_int_mod_prior(self, prior):
        mod_prior = 5.0 % prior
        assert mod_prior.instance_from_unit_vector(
            [3.0],
            assert_priors_in_limits=False
        ) == 2.0


def test_abs(prior):
    prior = af.UniformPrior(
        -1, 0
    )
    assert prior.value_for(0.0) == -1
    prior = abs(
        prior
    )
    assert prior.instance_from_unit_vector(
        [0.0]
    ) == 1.0


class TestPowers:
    def test_prior_to_prior(self, prior):
        power_prior = prior ** prior
        assert power_prior.instance_from_unit_vector(
            [2.0],
            assert_priors_in_limits=False
        ) == 4.0

    def test_prior_to_float(self, prior):
        power_prior = prior ** 3
        assert power_prior.instance_from_unit_vector(
            [2.0],
            assert_priors_in_limits=False
        ) == 8.0

    def test_float_to_prior(self, prior):
        power_prior = 3.0 ** prior
        assert power_prior.instance_from_unit_vector(
            [2.0],
            assert_priors_in_limits=False
        ) == 9.0


class TestInequality:
    def test_prior_lt_prior(self, prior):
        inequality_prior = (prior * prior) < prior
        assert inequality_prior.instance_from_unit_vector(
            [0.5],
            assert_priors_in_limits=False
        ) is True
        inequality_prior = (prior * prior) > prior
        assert inequality_prior.instance_from_unit_vector(
            [0.5],
            assert_priors_in_limits=False
        ) is False
