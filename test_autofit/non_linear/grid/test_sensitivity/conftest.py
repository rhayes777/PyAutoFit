import numpy as np
import pytest
from typing import Optional

import autofit as af
from autofit.non_linear.grid import sensitivity as s

x = np.array(range(10))

class Simulate:

    def __init__(self):

        pass

    def __call__(self, instance: af.ModelInstance, simulate_path : Optional[str]):

        image = instance.gaussian(x)

        if hasattr(instance, "perturbation"):
            image += instance.perturbation(x)

        return image


class Analysis(af.Analysis):

    def __init__(self, image: np.array):
        self.image = image

    def log_likelihood_function(self, instance):

        simulate = Simulate()

        image = simulate(instance, simulate_path=None)

        return np.mean(np.multiply(-0.5, np.square(np.subtract(self.image, image))))


@pytest.fixture(
    name="perturb_model"
)
def make_perturb_model():
    return af.Model(af.Gaussian)


@pytest.fixture(
    name="search"
)
def make_search():
    return af.m.MockSearch(return_sensitivity_results=True)


@pytest.fixture(
    name="sensitivity"
)
def make_sensitivity(
        perturb_model,
        search
):
    # noinspection PyTypeChecker
    instance = af.ModelInstance()
    instance.gaussian = af.Gaussian()
    return s.Sensitivity(
        simulation_instance=instance,
        base_model=af.Collection(
            gaussian=af.Model(af.Gaussian)
        ),
        perturb_model=perturb_model,
        simulate_cls=Simulate(),
        analysis_class=Analysis,
        search=search,
        number_of_steps=2,
    )


class MockAnalysisFactory:
    def __init__(self, analysis):
        self.analysis = analysis

    def __call__(self):
        return self.analysis


@pytest.fixture(
    name="job"
)
def make_job(
        perturb_model,
        search
):
    instance = af.ModelInstance()
    instance.gaussian = af.Gaussian()
    base_instance = instance
    instance.perturbation = af.Gaussian()
    image = Simulate()(instance, "")
    # noinspection PyTypeChecker
    return s.Job(
        model=af.Collection(
            gaussian=af.Model(af.Gaussian)
        ),
        perturb_model=af.Model(af.Gaussian),
        base_instance=base_instance,
        perturb_instance=instance,
        analysis_factory=MockAnalysisFactory(Analysis(image)),
        search=search,
        number=1
    )
