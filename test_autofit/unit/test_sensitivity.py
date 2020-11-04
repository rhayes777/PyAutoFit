import autofit as af
from autofit import sensitivity as s
from autofit.mock import mock


def test_lists():
    prior_model = af.PriorModel(
        mock.Gaussian
    )
    sensitivity = s.Sensitivity(
        perturbation_model=prior_model
    )
    assert len(list(sensitivity.perturbation_instances)) == 1000
