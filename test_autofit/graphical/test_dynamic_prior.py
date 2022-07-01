import autofit as af
from autofit import graphical as g


def test_power():
    prior = af.GaussianPrior(mean=1.0, sigma=2.0)
    mean_field = g.MeanField({
        prior: prior.message
    })
    new = mean_field ** g.MeanField({
        prior: 1.0
    })
    assert new[prior].sigma == 2.0
    assert new[prior].mean == 1.0
