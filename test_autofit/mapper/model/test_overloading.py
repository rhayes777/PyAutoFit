import autofit as af

from autofit.mock.mock_model import MockOverload


def test_constructor():
    prior_model = af.PriorModel(MockOverload)

    assert prior_model.prior_count == 1

    instance = prior_model.instance_from_prior_medians()

    assert instance.one == 1.0
    assert instance.two == 2


def test_alternative():
    prior_model = af.PriorModel(MockOverload.with_two)

    assert prior_model.prior_count == 1

    instance = prior_model.instance_from_prior_medians()

    assert instance.two == 1.0
    assert instance.one == 1.0 / 2
