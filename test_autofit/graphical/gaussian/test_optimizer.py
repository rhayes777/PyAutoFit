import pytest

import autofit.graphical as g
from test_autofit.graphical.gaussian.model import Analysis


def test_gaussian(x, y, prior_model):
    factor_model = g.ModelFactor(
        prior_model,
        analysis=Analysis(
            x=x,
            y=y
        )
    )

    laplace = g.LaplaceFactorOptimiser()
    model = factor_model.optimise(laplace)

    assert model.centre.mean == pytest.approx(50, rel=0.1)
    assert model.intensity.mean == pytest.approx(25, rel=0.1)
    assert model.sigma.mean == pytest.approx(10, rel=0.1)
