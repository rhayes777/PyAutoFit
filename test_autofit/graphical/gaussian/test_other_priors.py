import numpy as np
import pytest

import autofit as af
import autofit.graphical as ep
from test_autofit.graphical.gaussian.model import Gaussian, make_data, Analysis


@pytest.fixture(
    name="make_model_factor"
)
def make_make_model_factor(
        intensity,
        intensity_prior,
        x
):
    def make_factor_model(
            centre: float,
            sigma: float,
            optimiser=None
    ) -> ep.AnalysisFactor:
        y = make_data(
            Gaussian(
                centre=centre,
                intensity=intensity,
                sigma=sigma
            ),
            x
        )

        prior_model = af.PriorModel(
            Gaussian,
            centre=af.UniformPrior(lower_limit=10, upper_limit=100),
            intensity=intensity_prior,
            sigma=af.UniformPrior(lower_limit=0, upper_limit=20),
        )

        return ep.AnalysisFactor(
            prior_model,
            analysis=Analysis(
                x=x,
                y=y
            ),
            optimiser=optimiser
        )

    return make_factor_model


@pytest.fixture(
    name="intensity"
)
def make_intensity():
    return 25.0


@pytest.fixture(
    name="intensity_prior"
)
def make_intensity_prior():
    return af.UniformPrior(lower_limit=15, upper_limit=35)


@pytest.fixture(
    name="factor_model"
)
def make_factor_model_collection(
        make_model_factor
):
    return ep.FactorGraphModel(
        make_model_factor(
            centre=40,
            sigma=10
        ),
        make_model_factor(
            centre=60,
            sigma=15
        )
    )


def test_uniform_edge():
    uniform_prior = af.UniformPrior(
        lower_limit=10,
        upper_limit=20
    )
    assert not np.isnan(uniform_prior.logpdf(10))
    assert not np.isnan(uniform_prior.logpdf(20))


def _test_optimise_factor_model(
        factor_model
):
    laplace = ep.LaplaceFactorOptimiser()

    collection = factor_model.optimise(laplace)

    assert 25.0 == pytest.approx(collection[0].intensity.mean, rel=0.1)
    assert collection[0].intensity is collection[1].intensity


def test_trivial():
    uniform_prior = af.UniformPrior(
        lower_limit=10,
        upper_limit=20
    )

    # assert uniform_prior.value_for(0.5) is not None

    # x = list(range(30))
    # y = list(map(uniform_prior.logpdf, x))
    #
    # plt.plot(x, y)
    # plt.show()

    prior_model = af.Collection(
        value=uniform_prior
    )

    class TrivialAnalysis(af.Analysis):
        def log_likelihood_function(self, instance):
            result = -10e10 * (instance.value - 14) ** 2
            print(f"analysis: {instance.value} -> {result}")
            return result

    factor_model = ep.AnalysisFactor(
        prior_model,
        analysis=TrivialAnalysis()
    )

    # optimiser = ep.LaplaceFactorOptimiser()
    optimiser = af.DynestyStatic(maxcall=10)
    model = factor_model.optimise(
        optimiser
    )

    assert model.value.mean == 14


def test_gaussian():
    n_observations = 100
    x = np.arange(n_observations)
    y = make_data(Gaussian(centre=50.0, intensity=25.0, sigma=10.0), x)

    prior_model = af.PriorModel(
        Gaussian,
        centre=af.UniformPrior(lower_limit=30, upper_limit=70),
        intensity=af.UniformPrior(lower_limit=15, upper_limit=35),
        sigma=af.UniformPrior(lower_limit=0, upper_limit=20),
    )

    factor_model = ep.AnalysisFactor(
        prior_model,
        analysis=Analysis(
            x=x,
            y=y
        )
    )

    laplace = ep.LaplaceFactorOptimiser()
    model = factor_model.optimise(laplace)

    assert model.centre.mean == pytest.approx(50, rel=0.1)
    assert model.intensity.mean == pytest.approx(25, rel=0.1)
    assert model.sigma.mean == pytest.approx(10, rel=0.1)
