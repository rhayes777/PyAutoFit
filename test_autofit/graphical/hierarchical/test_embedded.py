from random import seed

import matplotlib.pyplot as plt
import numpy as np
import pytest

import autofit as af
from autofit import graphical as g
from test_autofit.graphical.gaussian.model import Analysis

x = np.arange(200)
n = 1

should_plot = False


@pytest.fixture(name="centre_model")
def make_centre_model():
    return g.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(mean=100, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=5),
    )


def test_embedded_priors(centre_model):
    assert isinstance(centre_model.random_instance().value_for(0.5), float)


def test_hierarchical_factor(centre_model):
    centre_model.add_drawn_variable(af.GaussianPrior(100, 10))
    factor = centre_model.factors[0]

    assert len(factor.priors) == 3

    laplace = g.LaplaceOptimiser()

    gaussian = factor.optimise(laplace, max_steps=10)
    assert gaussian.instance_from_prior_medians().drawn_prior.mean() == pytest.approx(
        100, abs=1
    )


@pytest.fixture(name="centres")
def make_centre(centre_model):
    seed(1)
    centres = list()
    for _ in range(n):
        centres.append(centre_model.random_instance().value_for(0.5))
    return centres


@pytest.fixture(name="data")
def generate_data(centres):
    data = []
    for centre in centres:
        gaussian = af.Gaussian(
            centre=centre,
            normalization=20,
            sigma=5,
        )

        data.append(gaussian(x))
    return data


def test_generate_data(data):
    if should_plot:
        for gaussian in data:
            plt.plot(x, gaussian)
        plt.show(block=False)


def test_model_factor(data, centres):
    y = data[0]
    centre_argument = af.GaussianPrior(mean=50, sigma=20)
    prior_model = af.PriorModel(
        af.Gaussian, centre=centre_argument, normalization=20, sigma=5
    )
    factor = g.AnalysisFactor(prior_model, analysis=Analysis(x=x, y=y))
    laplace = g.LaplaceOptimiser()

    gaussian = factor.optimise(laplace, max_steps=10)
    assert gaussian.centre.mean == pytest.approx(centres[0], abs=0.1)


def test_full_fit(centre_model, data, centres):
    graph = g.FactorGraphModel()
    for i, y in enumerate(data):
        prior_model = af.PriorModel(
            af.Gaussian,
            centre=af.GaussianPrior(mean=100, sigma=20),
            normalization=20,
            sigma=5,
        )
        graph.add(g.AnalysisFactor(prior_model, analysis=Analysis(x=x, y=y)))
        centre_model.add_drawn_variable(prior_model.centre)

    graph.add(centre_model)

    optimiser = g.LaplaceOptimiser()

    collection = graph.optimise(optimiser, max_steps=10).model

    # TODO I don't know what's going on here?
    # pred_centre = (
    #     collection.HierarchicalFactor0.distribution_model.instance_from_prior_medians().mean
    # )
    # (centre,) = centres
    # pred_centre == pytest.approx(centre, rel=0.1)

    # for gaussian, centre in zip(collection.with_prefix("AnalysisFactor"), centres):
    #     assert gaussian.instance_from_prior_medians().centre == pytest.approx(
    #         centre, abs=0.1
    #     )
