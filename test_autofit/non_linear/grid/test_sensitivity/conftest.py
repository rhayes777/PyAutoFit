import numpy as np
import pytest
from typing import Optional

import autofit as af
from autofit.non_linear.grid import sensitivity as s
from autofit.non_linear.mock.mock_samples_summary import MockSamplesSummary

x = np.array(range(10))


class Simulate:
    def __init__(self):
        pass

    def __call__(self, instance: af.ModelInstance, simulate_path: Optional[str]):
        image = instance.gaussian(x)

        if hasattr(instance, "perturb"):
            image += instance.perturb(x)

        return image


class Analysis(af.Analysis):
    def __init__(self, dataset: np.array):
        self.dataset = dataset

    def log_likelihood_function(self, instance):
        simulate = Simulate()

        dataset = simulate(instance, simulate_path=None)

        return np.mean(np.multiply(-0.5, np.square(np.subtract(self.dataset, dataset))))


class BaseFit:
    def __init__(self, analysis_cls):
        self.analysis_cls = analysis_cls

    def __call__(self, dataset, model, paths, instance):
        search = af.m.MockSearch(
            return_sensitivity_results=True,
            samples_summary=MockSamplesSummary(model=model),
            paths=paths,
        )

        analysis = self.analysis_cls(dataset=dataset)

        return search.fit(model=model, analysis=analysis)


class PerturbFit:
    def __init__(self, analysis_cls):
        self.analysis_cls = analysis_cls

    def __call__(self, dataset, model, paths, instance):
        search = af.m.MockSearch(
            return_sensitivity_results=True,
            samples_summary=MockSamplesSummary(model=model),
            paths=paths,
        )

        analysis = self.analysis_cls(dataset=dataset)

        return search.fit(model=model, analysis=analysis)


@pytest.fixture(name="perturb_model")
def make_perturb_model():
    return af.Model(af.Gaussian)


@pytest.fixture(name="sensitivity")
def make_sensitivity(
    perturb_model,
):
    # noinspection PyTypeChecker
    instance = af.ModelInstance()
    instance.gaussian = af.Gaussian()
    return s.Sensitivity(
        simulation_instance=instance,
        base_model=af.Collection(gaussian=af.Model(af.Gaussian)),
        perturb_model=perturb_model,
        simulate_cls=Simulate(),
        base_fit_cls=BaseFit(Analysis),
        perturb_fit_cls=PerturbFit(Analysis),
        paths=af.DirectoryPaths(),
        number_of_steps=2,
    )


@pytest.fixture(name="masked_sensitivity")
def make_masked_sensitivity(
    perturb_model,
):
    # noinspection PyTypeChecker
    instance = af.ModelInstance()
    instance.gaussian = af.Gaussian()
    return s.Sensitivity(
        simulation_instance=instance,
        base_model=af.Collection(gaussian=af.Model(af.Gaussian)),
        perturb_model=perturb_model,
        simulate_cls=Simulate(),
        base_fit_cls=BaseFit(Analysis),
        perturb_fit_cls=PerturbFit(Analysis),
        paths=af.DirectoryPaths(),
        number_of_steps=2,
        mask=np.array(
            [
                [
                    [True, True],
                    [True, True],
                ],
                [
                    [True, True],
                    [True, True],
                ],
            ]
        ),
    )


@pytest.fixture(name="job")
def make_job(
    perturb_model,
):
    instance = af.ModelInstance()
    instance.gaussian = af.Gaussian()
    base_instance = instance
    instance.perturb = af.Gaussian()
    # noinspection PyTypeChecker
    return s.Job(
        model=af.Collection(gaussian=af.Model(af.Gaussian)),
        perturb_model=af.Model(af.Gaussian),
        simulate_instance=instance,
        base_instance=base_instance,
        simulate_cls=Simulate(),
        base_fit_cls=BaseFit(Analysis),
        perturb_fit_cls=PerturbFit(Analysis),
        paths=af.DirectoryPaths(),
        number=1,
    )
