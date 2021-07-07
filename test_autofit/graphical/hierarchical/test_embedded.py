from random import seed

import matplotlib.pyplot as plt
import numpy as np
import pytest

import autofit as af
from autofit import graphical as g
from autofit.mock.mock import Gaussian
from test_autofit.graphical.gaussian.model import Analysis

x = np.arange(200)
n = 1

should_plot = False


@pytest.fixture(
    name="centre_model"
)
def make_centre_model():
    return af.PriorModel(
        af.GaussianPrior,
        mean=af.GaussianPrior(
            mean=100,
            sigma=10
        ),
        sigma=af.GaussianPrior(
            mean=10,
            sigma=5
        )
    )


def test_embedded_priors(
        centre_model
):
    assert isinstance(
        centre_model.random_instance().value_for(0.5),
        float
    )


def test_hierarchical_factor(
        centre_model
):
    factor = g.HierarchicalFactor(
        centre_model,
        af.GaussianPrior(100, 10)
    )

    assert len(factor.priors) == 3

    laplace = g.LaplaceFactorOptimiser()

    gaussian = factor.optimise(laplace, max_steps=10)
    assert gaussian.instance_from_prior_medians().mean == pytest.approx(100, abs=1)


@pytest.fixture(
    name="centres"
)
def make_centre(
        centre_model
):
    seed(1)
    centres = list()
    for _ in range(n):
        centres.append(
            centre_model.random_instance().value_for(0.5)
        )
    return centres


@pytest.fixture(
    name="data"
)
def generate_data(
        centres
):
    data = []
    for centre in centres:
        gaussian = Gaussian(
            centre=centre,
            intensity=20,
            sigma=5,
        )

        data.append(
            gaussian(x)
        )
    return data


def test_generate_data(
        data
):
    if should_plot:
        for gaussian in data:
            plt.plot(x, gaussian)
        plt.show(block=False)


def test_model_factor(
        data,
        centres
):
    y = data[0]
    centre_argument = af.GaussianPrior(
        mean=50,
        sigma=20
    )
    prior_model = af.PriorModel(
        Gaussian,
        centre=centre_argument,
        intensity=20,
        sigma=5
    )
    factor = g.AnalysisFactor(
        prior_model,
        analysis=Analysis(
            x=x,
            y=y
        )
    )
    laplace = g.LaplaceFactorOptimiser()

    gaussian = factor.optimise(laplace, max_steps=10)
    assert gaussian.centre.mean == pytest.approx(centres[0], abs=0.1)


def test_full_fit(centre_model, data, centres):
    graph = g.FactorGraphModel()
    for i, y in enumerate(data):
        centre_argument = af.GaussianPrior(
            mean=100,
            sigma=20
        )
        prior_model = af.PriorModel(
            Gaussian,
            centre=centre_argument,
            intensity=20,
            sigma=5
        )
        graph.add(
            g.AnalysisFactor(
                prior_model,
                analysis=Analysis(
                    x=x,
                    y=y
                )
            )
        )
        graph.add(
            g.HierarchicalFactor(
                centre_model,
                centre_argument
            )
        )

    laplace = g.LaplaceFactorOptimiser()

    collection = graph.optimise(laplace, max_steps=10)

    for gaussian, centre in zip(
            collection.with_prefix(
                "AnalysisFactor"
            ),
            centres
    ):
        assert gaussian.instance_from_prior_medians().centre == pytest.approx(
            centre,
            abs=0.1
        )
