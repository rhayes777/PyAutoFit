import autofit as af
from autofit.mock.mock import Gaussian


def test_freeze():
    model = af.PriorModel(
        Gaussian
    )

    model.freeze()

    model.instance_from_prior_medians()
