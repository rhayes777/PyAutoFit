import autofit as af
from autofit.mock import mock as m


def test_simple():
    target = af.PriorModel(
        m.Gaussian
    )

    prior = af.UniformPrior()
    source = af.PriorModel(
        m.Gaussian,
        centre=prior
    )

    target.take_attributes(
        source
    )

    assert target.centre is prior
