import math

import pytest

import autofit as af


@pytest.fixture(name="prior")
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
        assert (
                division_prior.instance_from_unit_vector(
                    [0.5], ignore_prior_limits=True
                )
                == 1
        )

    def test_prior_over_float(self, prior):
        division_prior = prior / 2
        assert division_prior.instance_from_unit_vector([1.0]) == 0.5

    def test_float_over_prior(self, prior):
        division_prior = 4.0 / prior
        assert division_prior.instance_from_unit_vector([0.5]) == 8.0


@pytest.fixture(
    name="ten_prior"
)
def make_ten_prior():
    return af.UniformPrior(
        lower_limit=0.0,
        upper_limit=10.0
    )


class TestFloorDiv:
    def test_prior_over_int(self, ten_prior):
        division_prior = ten_prior // 2
        assert (
                division_prior.instance_from_unit_vector(
                    [0.5], ignore_prior_limits=True
                )
                == 2.0
        )

    def test_int_over_prior(self, ten_prior):
        division_prior = 3 // ten_prior
        assert (
                division_prior.instance_from_unit_vector(
                    [0.2], ignore_prior_limits=True
                )
                == 1.0
        )


class TestMod:
    def test_prior_mod_int(self, ten_prior):
        mod_prior = ten_prior % 3
        assert (
                mod_prior.instance_from_unit_vector([0.5], ignore_prior_limits=True)
                == 2.0
        )

    def test_int_mod_prior(self, ten_prior):
        mod_prior = 5.0 % ten_prior
        assert (
                mod_prior.instance_from_unit_vector([0.3], ignore_prior_limits=True)
                == 2.0
        )


def test_abs(prior):
    prior = af.UniformPrior(-1, 0)
    assert prior.value_for(0.0) == -1
    prior = abs(prior)
    assert prior.instance_from_unit_vector([0.0]) == 1.0


class TestPowers:
    def test_prior_to_prior(self, ten_prior):
        power_prior = ten_prior ** ten_prior
        assert (
                power_prior.instance_from_unit_vector([0.2], ignore_prior_limits=True)
                == 4.0
        )

    def test_prior_to_float(self, ten_prior):
        power_prior = ten_prior ** 3
        assert (
                power_prior.instance_from_unit_vector([0.2], ignore_prior_limits=True)
                == 8.0
        )

    def test_float_to_prior(self, ten_prior):
        power_prior = 3.0 ** ten_prior
        assert (
                power_prior.instance_from_unit_vector([0.2], ignore_prior_limits=True)
                == 9.0
        )


class TestInequality:
    def test_prior_lt_prior(self, prior):
        inequality_prior = (prior * prior) < prior
        result = inequality_prior.instance_from_unit_vector(
            [0.5], ignore_prior_limits=True
        )
        assert result
        inequality_prior = (prior * prior) > prior
        assert not (
            inequality_prior.instance_from_unit_vector(
                [0.5], ignore_prior_limits=True
            )
        )


@pytest.mark.parametrize(
    "multiplier, value",
    [
        (math.e, 1),
        (math.e ** 2, 2),
        (1, 0)
    ]
)
def test_log(
        multiplier,
        value,
        prior
):
    assert af.Log(
        multiplier * prior
    ).instance_from_unit_vector(
        [1.0]
    ) == pytest.approx(value)


@pytest.mark.parametrize(
    "multiplier, value",
    [
        (10, 1),
        (1, 0),
        (100, 2),
        (1000, 3),
    ]
)
def test_log_10(
        multiplier,
        value,
        prior
):
    assert af.Log10(
        multiplier * prior
    ).instance_from_unit_vector(
        [1.0]
    ) == pytest.approx(value)
