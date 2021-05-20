import pytest

import autofit.graphical as g
from test_autofit.graphical.gaussian.model import Analysis


@pytest.fixture(
    name="factor_model"
)
def make_factor_model(
        prior_model,
        x,
        y
):
    return g.ModelFactor(
        prior_model,
        analysis=Analysis(
            x=x,
            y=y
        )
    )


def test_gaussian(factor_model):
    laplace = g.LaplaceFactorOptimiser()
    model = factor_model.optimise(laplace)

    assert model.centre.mean == pytest.approx(50, rel=0.1)
    assert model.intensity.mean == pytest.approx(25, rel=0.1)
    assert model.sigma.mean == pytest.approx(10, rel=0.1)
