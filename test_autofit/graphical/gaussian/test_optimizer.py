import pytest

import autofit as af
import autofit.graphical as g
from autoconf.conf import output_path_for_test
from autofit.graphical.utils import Status
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


@pytest.fixture(
    name="laplace"
)
def make_laplace():
    return g.LaplaceFactorOptimiser()


@pytest.fixture(
    name="dynesty"
)
def make_dynesty():
    return af.DynestyStatic(
        maxcall=10
    )


def test_default(
        factor_model,
        laplace
):
    model = factor_model.optimise(laplace)

    assert model.centre.mean == pytest.approx(50, rel=0.1)
    assert model.intensity.mean == pytest.approx(25, rel=0.1)
    assert model.sigma.mean == pytest.approx(10, rel=0.1)


class TestDynesty:
    @output_path_for_test()
    def test_optimisation(
            self,
            factor_model,
            laplace,
            dynesty
    ):
        factor_model.optimiser = dynesty
        factor_model.optimise(laplace)

    @output_path_for_test()
    def test_optimise(
            self,
            factor_model,
            dynesty
    ):
        result, status = dynesty.optimise(
            factor_model,
            factor_model.mean_field_approximation()
        )

        assert isinstance(
            result,
            g.EPMeanField
        )
        assert isinstance(
            status,
            Status
        )
