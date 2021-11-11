import pytest

from autofit import graphical as g
from autofit.graphical.expectation_propagation import FactorHistory


@pytest.fixture(
    name="success"
)
def make_success():
    return g.Status(
        success=True
    )


@pytest.fixture(
    name="failure"
)
def make_failure():
    return g.Status(
        success=False
    )


def test_truthy_status(
        success,
        failure
):
    assert success
    assert not failure


def test_latest(
        success
):
    factor = g.Factor(sum)
    approx = g.EPMeanField(
        factor_graph=g.FactorGraph([
            factor
        ]),
        factor_mean_field={
            factor: g.MeanField({})
        }
    )
    factor_history = FactorHistory(
        g.Factor(sum)
    )
    factor_history(
        approx, success
    )

    assert factor_history.latest_successful == approx
