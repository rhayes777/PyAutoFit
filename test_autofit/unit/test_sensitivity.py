import numpy as np
import pytest

import autofit as af
from autofit import sensitivity as s
from autofit.mock.mock import Gaussian


@pytest.fixture(
    name="perturbation_model"
)
def make_perturbation_model():
    return af.PriorModel(
        Gaussian
    )


x = np.array(range(10))


def image_function(instance, perturbation_instance):
    return instance(x) + perturbation_instance(x)


def test_lists(perturbation_model):
    sensitivity = s.Sensitivity(
        perturbation_model=perturbation_model,
        image_function=lambda: None
    )
    assert len(list(sensitivity.perturbation_instances)) == 1000


def test_job(perturbation_model):
    job = s.Job(
        instance=Gaussian(),
        model=af.PriorModel(Gaussian),
        perturbation_instance=Gaussian(),
        perturbation_model=af.PriorModel(Gaussian),
        image_function=image_function
    )
    result = job.perform()
    assert isinstance(
        result,
        s.JobResult
    )
