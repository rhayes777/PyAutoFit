import numpy as np
import pytest

import autofit as af
import autofit.graphical as ep
from test_autofit.graphical.gaussian.model import Gaussian, make_data, Analysis


@pytest.fixture(name="make_model_factor")
def make_make_model_factor(normalization, normalization_prior, x):
    def make_factor_model(
            centre: float, sigma: float, optimiser=None
    ) -> ep.AnalysisFactor:
        y = make_data(
            Gaussian(centre=centre, normalization=normalization, sigma=sigma), x
        )

        prior_model = af.PriorModel(
            Gaussian,
            centre=af.UniformPrior(lower_limit=10, upper_limit=100),
            normalization=normalization_prior,
            sigma=af.UniformPrior(lower_limit=0, upper_limit=20),
        )

        return ep.AnalysisFactor(
            prior_model, analysis=Analysis(x=x, y=y), optimiser=optimiser
        )

    return make_factor_model


@pytest.fixture(name="normalization")
def make_normalization():
    return 25.0


@pytest.fixture(name="normalization_prior")
def make_normalization_prior():
    return af.UniformPrior(lower_limit=15, upper_limit=35)


@pytest.fixture(name="factor_model")
def make_factor_model_collection(make_model_factor):
    return ep.FactorGraphModel(
        make_model_factor(centre=40, sigma=10), make_model_factor(centre=60, sigma=15)
    )


def test_uniform_edge():
    uniform_prior = af.UniformPrior(lower_limit=10, upper_limit=20)
    assert not np.isnan(uniform_prior.logpdf(10))
    assert not np.isnan(uniform_prior.logpdf(20))


def _test_optimise_factor_model(factor_model):
    laplace = ep.LaplaceOptimiser()

    collection = factor_model.optimise(laplace)

    assert 25.0 == pytest.approx(collection[0].normalization.mean, rel=0.1)
    assert collection[0].normalization is collection[1].normalization


def test_trivial():
    prior = af.UniformPrior(lower_limit=10, upper_limit=20)

    prior_model = af.Collection(value=prior)

    class TrivialAnalysis(af.Analysis):
        def log_likelihood_function(self, instance):
            result = -((instance.value - 14) ** 2)
            return result

    factor_model = ep.AnalysisFactor(prior_model, analysis=TrivialAnalysis())

    optimiser = ep.LaplaceOptimiser()
    # optimiser = af.DynestyStatic()
    model = factor_model.optimise(optimiser)

    assert model.value.mean == pytest.approx(14, rel=0.1)


def _test_gaussian():
    n_observations = 100
    x = np.arange(n_observations)
    y = make_data(Gaussian(centre=50.0, normalization=25.0, sigma=10.0), x)

    prior_model = af.PriorModel(
        Gaussian,
        # centre=af.GaussianPrior(mean=50, sigma=10),
        # normalization=af.GaussianPrior(mean=25, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=10),
        centre=af.UniformPrior(lower_limit=30, upper_limit=70),
        normalization=af.UniformPrior(lower_limit=15, upper_limit=35),
        # sigma=af.UniformPrior(lower_limit=5, upper_limit=15),
    )

    factor_model = ep.AnalysisFactor(prior_model, analysis=Analysis(x=x, y=y))

    # optimiser = ep.LaplaceOptimiser(
    #     transform_cls=DiagonalMatrix
    # )
    optimiser = af.DynestyStatic()
    model = factor_model.optimise(optimiser)

    assert model.centre.mean == pytest.approx(50, rel=0.1)
    assert model.normalization.mean == pytest.approx(25, rel=0.1)
    assert model.sigma.mean == pytest.approx(10, rel=0.1)
