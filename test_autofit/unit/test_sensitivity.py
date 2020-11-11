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


@pytest.fixture(
    name="sensitivity"
)
def make_sensitivity(perturbation_model):
    return s.Sensitivity(
        instance=Gaussian(),
        model=af.PriorModel(Gaussian),
        perturbation_model=perturbation_model,
        image_function=image_function,
        step_size=0.5
    )


x = np.array(range(10))


def image_function(instance, perturbation_instance):
    return instance(x) + perturbation_instance(x)


def test_lists(sensitivity):
    assert len(list(sensitivity.perturbation_instances)) == 8


def test_sensitivity(sensitivity):
    results = sensitivity.run()
    assert len(results) == 8


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
