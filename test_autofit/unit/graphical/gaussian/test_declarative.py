import numpy as np
import pytest

import autofit as af
import autofit.graphical as ep
from test_autofit.unit.graphical.gaussian.model import Gaussian, make_data, Analysis


def test_shared_intensity():
    """
    Here's a good example in which we have two Gaussians fit with a shared variable
    """
    n_observations = 100
    x = np.arange(n_observations)

    """
    We have a shared intensity value and a shared intensity prior
    """
    intensity = 25.0
    intensity_prior = af.GaussianPrior(mean=25, sigma=10)

    def make_factor_model(centre: float, sigma: float) -> ep.ModelFactor:
        """
        We'll make a LikelihoodModel for each Gaussian we're fitting.

        First we'll make the actual data to be fit.

        Note that the intensity value is shared.
        """
        y = make_data(Gaussian(centre=centre, intensity=intensity, sigma=sigma), x)

        """
        Next we need a prior model.
        
        Note that the intensity prior is shared.
        """
        prior_model = af.PriorModel(
            Gaussian,
            centre=af.GaussianPrior(mean=50, sigma=20),
            intensity=intensity_prior,
            sigma=af.GaussianPrior(mean=10, sigma=10),
        )

        """
        Finally we combine the likelihood function with the prior model to produce a likelihood
        factor - this will be converted into a ModelFactor which is like any other factor in the
        factor graph.
        """
        return ep.ModelFactor(
            prior_model,
            analysis=Analysis(
                x=x,
                y=y
            )
        )

    """
    Multiplying together multiple LikelihoodModels gives us a factor model.
    
    The factor model can compute all the variables and messages required as well as construct
    a factor graph representing a fit on the ensemble.
    """
    factor_model = ep.GraphicalModel(
        make_factor_model(centre=40, sigma=10), make_factor_model(centre=60, sigma=15)
    )

    """
    There are:
    - 5 messages - one for each prior 
    - 7 factors - one for each prior plus one for each likelihood
    """
    assert len(factor_model.message_dict) == 5
    assert len(factor_model.graph.factors) == 7

    """
    We optimise that...
    """
    opt = ep.optimise.LaplaceOptimiser(n_iter=3)
    collection = factor_model.optimise(opt)

    """
    And what we get back is actually a PriorModelCollection
    """
    assert 25.0 == pytest.approx(collection[0].intensity.mean, rel=0.1)
    assert collection[0].intensity is collection[1].intensity


def test_gaussian():
    n_observations = 100
    x = np.arange(n_observations)
    y = make_data(Gaussian(centre=50.0, intensity=25.0, sigma=10.0), x)

    prior_model = af.PriorModel(
        Gaussian,
        centre=af.GaussianPrior(mean=50, sigma=20),
        intensity=af.GaussianPrior(mean=25, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=10),
    )

    factor_model = ep.ModelFactor(
        prior_model,
        analysis=Analysis(
            x=x,
            y=y
        )
    )

    opt = ep.optimise.LaplaceOptimiser(n_iter=3)
    model = factor_model.optimise(opt)

    assert model.centre.mean == pytest.approx(50, rel=0.1)
    assert model.intensity.mean == pytest.approx(25, rel=0.1)
    assert model.sigma.mean == pytest.approx(10, rel=0.1)


@pytest.fixture(name="prior_model")
def make_prior_model():
    return af.PriorModel(Gaussian)


@pytest.fixture(name="likelihood_model")
def make_factor_model(prior_model):
    class MockAnalysis(af.Analysis):
        @staticmethod
        def log_likelihood_function(*_):
            return 1

    return ep.ModelFactor(
        prior_model,
        analysis=MockAnalysis()
    )


def test_messages(likelihood_model):
    assert len(likelihood_model.message_dict) == 3


def test_graph(likelihood_model):
    graph = likelihood_model.graph
    assert len(graph.factors) == 4


def test_prior_model_node(likelihood_model):
    prior_model_node = likelihood_model.graph

    result = prior_model_node(
        {variable: np.array([0.5]) for variable in prior_model_node.variables}
    )

    assert isinstance(result, ep.FactorValue)
