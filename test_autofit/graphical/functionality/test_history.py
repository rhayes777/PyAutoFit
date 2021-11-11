from copy import copy

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


@pytest.fixture(
    name="factor"
)
def make_factor():
    return g.Factor(sum)


@pytest.fixture(
    name="approx"
)
def make_approx(factor):
    return g.EPMeanField(
        factor_graph=g.FactorGraph([
            factor
        ]),
        factor_mean_field={
            factor: g.MeanField({})
        }
    )


@pytest.fixture(
    name="factor_history"
)
def make_factor_history(
        factor,
        approx,
        success
):
    return FactorHistory(
        factor
    )


def test_latest(
        success,
        factor_history,
        approx
):
    factor_history(
        approx, success
    )

    assert factor_history.latest_successful == approx


def test_previous(
        success,
        factor_history,
        approx
):
    factor_history(
        approx, success
    )
    factor_history(
        copy(approx), success
    )

    assert factor_history.previous_successful is approx


def test_failure(
        success,
        failure,
        factor_history,
        approx
):
    approx_2 = copy(approx)

    factor_history(
        approx, success
    )
    factor_history(
        approx_2, success
    )
    factor_history(
        copy(approx), failure
    )

    assert factor_history.latest_successful == approx_2
    assert factor_history.previous_successful == approx


def test_kl_divergence(
        factor_history,
        approx,
        success
):
    factor_history(
        approx, success
    )
    factor_history(
        approx, success
    )
    assert factor_history.kl_divergence() == 0
