"""
A non-finite term in the EP log evidence is reported by name, not summed (PyAutoFit#1713).

On the analytic_gaussian ensemble (seeds 25 and 172, default ``projection="mode"``)
one member of a ``HierarchicalFactor`` sends the parent scatter a truncated-normal
message centred thousands of sigma below its support (mean -7374, sigma 384 on
[0, 100]). Its truncation mass ``Phi(b) - Phi(a)`` evaluates to ``1 - 1 = 0``, so its
log-partition is ``-inf``: the scatter's variable evidence is ``+inf``, every adjacent
factor term is ``-inf``, and ``EPMeanField.log_evidence`` added them, emitting numpy's
``invalid value encountered in add`` once per evaluation and returning NaN. The
posteriors are unaffected; only the evidence readout is.

The fixture below is that state in miniature: two factors share ``x``, one message is
the production message.
"""

import logging
import warnings

import numpy as np
import pytest

from autofit import graphical as g
from autofit.graphical.expectation_propagation import ep_mean_field
from autofit.graphical.expectation_propagation.diagnostics import (
    EPDiagnostics,
    check_evidence_nonfinite,
)
from autofit.graphical.expectation_propagation.history import FactorHistory
from autofit.mapper.variable import Variable
from autofit.messages.truncated_normal import TruncatedNormalMessage

LOGGER = "autofit.graphical.expectation_propagation.ep_mean_field"


def _ok(x):
    return x


def _degenerate(x):
    return -x


def make_approx(x, bad_mean=-7374.0, bad_sigma=384.0):
    ok_factor = g.Factor(_ok, x, name="f_ok")
    degenerate_factor = g.Factor(_degenerate, x, name="f_degenerate")
    ok = TruncatedNormalMessage(mean=10.0, sigma=5.0, lower_limit=0.0, upper_limit=100.0)
    bad = TruncatedNormalMessage(
        mean=bad_mean, sigma=bad_sigma, lower_limit=0.0, upper_limit=100.0
    )
    return g.EPMeanField(
        factor_graph=g.FactorGraph([ok_factor, degenerate_factor]),
        factor_mean_field={
            ok_factor: g.MeanField({x: ok}),
            degenerate_factor: g.MeanField({x: bad}),
        },
    )


@pytest.fixture(name="x")
def make_x():
    return Variable("x")


@pytest.fixture(name="degenerate")
def make_degenerate(x):
    return make_approx(x)


@pytest.fixture(name="healthy")
def make_healthy(x):
    return make_approx(x, bad_mean=20.0, bad_sigma=8.0)


def test_log_evidence_nan_without_runtime_warning(degenerate):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        log_evidence = degenerate.log_evidence

    assert isinstance(log_evidence, float)
    assert np.isnan(log_evidence)


def test_one_named_warning_per_run(degenerate, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(3):
            degenerate.log_evidence
        # A fresh EPMeanField over the same graph (what each EP update makes)
        # does not repeat the warning.
        g.EPMeanField(
            factor_graph=degenerate.factor_graph,
            factor_mean_field=degenerate.factor_mean_field,
        ).log_evidence

    records = [r for r in caplog.records if r.name == LOGGER]
    assert len(records) == 1
    assert "'x'" in records[0].getMessage()
    assert "+inf" in records[0].getMessage()


def test_nonfinite_evidence_terms_names_the_variable(degenerate):
    assert degenerate.nonfinite_evidence_terms == {"variable 'x'": "+inf"}


def test_healthy_evidence_unchanged(healthy, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            log_evidence = healthy.log_evidence

    variable_evidence = {
        v: np.sum(logz) for v, logz in healthy.variable_evidence.items()
    }
    expected = sum(
        np.sum(meanfield.log_norm)
        - sum(variable_evidence[v] for v in factor.all_variables)
        for factor, meanfield in healthy.factor_mean_field.items()
    ) + sum(variable_evidence.values())

    assert np.isfinite(log_evidence)
    assert log_evidence == expected
    assert healthy.nonfinite_evidence_terms == {}
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_evidence_divergence_inf_on_nan(degenerate, healthy):
    factor = next(iter(healthy.factor_mean_field))
    history = FactorHistory(factor)
    history(healthy, g.Status(success=True))
    history(degenerate, g.Status(success=True))

    assert history.evidence_divergence() == float("inf")


def test_evidence_divergence_finite_when_healthy(healthy):
    factor = next(iter(healthy.factor_mean_field))
    history = FactorHistory(factor)
    history(healthy, g.Status(success=True))
    history(healthy, g.Status(success=True))

    assert history.evidence_divergence() == 0.0


def test_diagnostics_record_evidence_nonfinite(degenerate, healthy):
    diagnostics = EPDiagnostics()
    factor = next(iter(healthy.factor_mean_field))
    diagnostics.snapshot(factor, healthy, g.Status(success=True))
    assert check_evidence_nonfinite(diagnostics) == []

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        diagnostics.snapshot(factor, degenerate, g.Status(success=True))
        diagnostics.snapshot(factor, degenerate, g.Status(success=True))

    assert np.isnan(diagnostics.factor_rows[-1]["log_evidence"])
    lines = check_evidence_nonfinite(diagnostics)
    assert len(lines) == 1
    assert lines[0].startswith("evidence_nonfinite: ")
    assert "variable 'x' (+inf, first at step 1)" in lines[0]


@pytest.fixture(autouse=True)
def reset_reported():
    ep_mean_field._reported_nonfinite.clear()
    yield
    ep_mean_field._reported_nonfinite.clear()
