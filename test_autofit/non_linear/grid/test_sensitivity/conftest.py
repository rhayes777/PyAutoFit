import numpy as np
import pytest

import autofit as af
from autofit.mock.mock import Gaussian
from autofit.non_linear.grid import sensitivity as s

x = np.array(range(10))


def image_function(instance: af.ModelInstance):
    image = instance.gaussian(x)
    if hasattr(instance, "perturbation"):
        image += instance.perturbation(x)
    return image


class Analysis(af.Analysis):

    def __init__(self, image: np.array):
        self.image = image

    def log_likelihood_function(self, instance):
        image = image_function(instance)
        return np.mean(np.multiply(-0.5, np.square(np.subtract(self.image, image))))


@pytest.fixture(
    name="perturbation_model"
)
def make_perturbation_model():
    return af.PriorModel(Gaussian)


@pytest.fixture(
    name="search"
)
def make_search():
    return af.MockSearch()


@pytest.fixture(
    name="sensitivity"
)
def make_sensitivity(
        perturbation_model,
        search
):
    # noinspection PyTypeChecker
    instance = af.ModelInstance()
    instance.gaussian = Gaussian()
    return s.Sensitivity(
        simulation_instance=instance,
        base_model=af.Collection(
            gaussian=af.PriorModel(Gaussian)
        ),
        perturbation_model=perturbation_model,
        simulate_function=image_function,
        analysis_class=Analysis,
        search=search,
        number_of_steps=2,
    )


@pytest.fixture(
    name="job"
)
def make_job(
        perturbation_model,
        search
):
    instance = af.ModelInstance()
    instance.gaussian = Gaussian()
    instance.perturbation = Gaussian()
    image = image_function(instance)
    # noinspection PyTypeChecker
    return s.Job(
        model=af.Collection(
            gaussian=af.PriorModel(Gaussian)
        ),
        perturbation_model=af.PriorModel(Gaussian),
        analysis=Analysis(image),
        search=search,
    )
