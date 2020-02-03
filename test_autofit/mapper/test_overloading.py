import math

import autofit as af
from test_autofit import mock


def test_constructor():
    prior_model = af.Model(mock.Circle)

    assert prior_model.prior_count == 1

    instance = prior_model.instance_from_prior_medians()

    assert instance.radius == 1.0
    assert instance.circumference == 2 * math.pi


def test_alternative():
    prior_model = af.Model(mock.Circle.with_circumference)

    assert prior_model.prior_count == 1

    instance = prior_model.instance_from_prior_medians()

    assert instance.circumference == 1.0
    assert instance.radius == 1.0 / (2 * math.pi)
