"""
Golden ``samples.csv`` fixtures for the search-extensibility phase A3 samples adapter.

Each case under ``golden/`` holds a backend internal stored from a small real fit
(Emcee, DynestyStatic, Nautilus, BlackJAXNUTS, SMC), the fitted model, and the
``samples.csv`` / ``samples_info.json`` that ``samples_via_internal_from`` produced
from that internal on ``origin/main`` before the adapter landed (``golden/generate.py``).
Converting the same internal today must reproduce ``samples.csv``: the same header,
the same rows in the same order, and every parameter, log likelihood, log prior, log
posterior and weight equal to within floating-point roundoff, with no renormalisation
(decision D13).

The comparison is to ``CSV_RTOL`` rather than byte for byte because NumPy's
vectorised ``exp`` / ``log`` loops are dispatched per CPU (AVX2 vs AVX-512), so the
same NumPy version on a different runner can land one unit in the last place away:
on CI the dynesty and nautilus weights (and two nautilus log priors) differed from the
AVX2-generated fixtures by at most 1 ULP (relative 2.2e-16). Any renormalisation,
reordering or dropped row changes values by many orders of magnitude more.

``samples_info`` is compared exactly except for the convergence diagnostics BlackJAX
itself computes (ESS, bulk/tail ESS, rank-normalised R-hat), whose estimators change
between BlackJAX releases. Those are pinned three ways, so a wrong value or wrong
wiring cannot pass on shape alone (review 05 finding 7):

- by value against the fixture, to ``DIAGNOSTIC_RTOL`` (loose enough for an estimator
  revision, far tighter than any wiring error), finite and positive;
- exactly against the installed BlackJAX's own calculation on the stored chains
  (``test_diagnostics_are_the_installed_blackjax_reference``);
- by ownership, with the BlackJAX estimators replaced by deterministic doubles whose
  outputs must be forwarded unchanged (``test_diagnostics_forward_the_backend_outputs``).
"""
import csv
import importlib.util
import json

import numpy as np
import pytest

from test_autofit.non_linear.samples.golden import cases

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _skip_missing(name):
    missing = [
        module
        for module in cases.CASES[name][3]
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        pytest.skip(f"golden case {name} needs {missing}")


@pytest.fixture(autouse=True)
def reference_mode(monkeypatch):
    # The fixtures were converted outside test mode, where the MCMC burn-in and
    # thinning come from the auto-correlation times rather than fixed values.
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)


#: Relative tolerance on every numeric ``samples.csv`` entry: ~4500x the largest drift
#: measured across CI runners (2.2e-16), far below any change the adapter could make.
CSV_RTOL = 1e-12

#: ``samples_info`` keys computed by ``blackjax.diagnostics`` rather than by autofit.
BACKEND_DIAGNOSTIC_KEYS = (
    "ess_min",
    "ess_per_param",
    "ess_bulk_min",
    "ess_bulk_per_param",
    "ess_tail_min",
    "ess_tail_per_param",
    "rhat_max",
    "rhat_per_param",
)


#: Relative tolerance on the BlackJAX diagnostics against the fixture: room for an
#: estimator revision between BlackJAX releases, while a sentinel, a swapped key or a
#: per-parameter mix-up moves them by far more.
DIAGNOSTIC_RTOL = 0.25


def assert_diagnostics_match(info: dict, expected: dict):
    """
    Pop every ``BACKEND_DIAGNOSTIC_KEYS`` entry from ``info`` and ``expected`` and
    compare them by value: the same shape, finite, positive (ESS and R-hat both are) and
    within ``DIAGNOSTIC_RTOL`` of the fixture.
    """
    for key in BACKEND_DIAGNOSTIC_KEYS:
        if key not in expected:
            continue
        actual = np.asarray(info.pop(key), dtype=float)
        reference = np.asarray(expected.pop(key), dtype=float)

        assert actual.shape == reference.shape, key
        assert np.all(np.isfinite(actual)), key
        assert np.all(actual > 0.0), key
        np.testing.assert_allclose(
            actual, reference, rtol=DIAGNOSTIC_RTOL, atol=0.0, err_msg=key
        )


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [column.strip() for column in rows[0]]
    values = np.array([[float(value) for value in row] for row in rows[1:]])
    return header, values


@pytest.mark.parametrize("name", list(cases.CASES))
def test_samples_csv_is_unchanged(name, tmp_path):
    _skip_missing(name)

    samples = cases.samples_from(name)
    samples.write_table(filename=tmp_path / "samples.csv")

    header, values = _read_csv(tmp_path / "samples.csv")
    expected_header, expected_values = _read_csv(cases.HERE / name / "samples.csv")

    assert header == expected_header
    assert values.shape == expected_values.shape
    np.testing.assert_allclose(values, expected_values, rtol=CSV_RTOL, atol=0.0)


@pytest.mark.parametrize("name", list(cases.CASES))
def test_samples_info_is_unchanged(name):
    _skip_missing(name)

    samples = cases.samples_from(name)

    with open(cases.HERE / name / "samples_info.json") as f:
        expected = json.load(f)

    info = cases.samples_info_for_comparison(samples)

    assert_diagnostics_match(info, expected)

    assert info == expected


# Review 05 finding 7: the diagnostics must be protected by value, not only by shape.


def _sentinel(value):
    return (np.full(np.shape(value), -999.0)).tolist() if np.ndim(value) else -999.0


@pytest.mark.parametrize("key", BACKEND_DIAGNOSTIC_KEYS)
def test_sentinel_diagnostics_fail_the_golden_comparison(key, monkeypatch):
    """
    Replacing any backend diagnostic with a shape-preserving ``-999`` must fail
    ``test_samples_info_is_unchanged``.
    """
    _skip_missing("nuts")

    original = cases.samples_info_for_comparison

    def mutated(samples):
        info = original(samples)
        info[key] = _sentinel(info[key])
        return info

    monkeypatch.setattr(cases, "samples_info_for_comparison", mutated)

    with pytest.raises(AssertionError):
        test_samples_info_is_unchanged("nuts")


def test_diagnostics_are_the_installed_blackjax_reference():
    """
    The NUTS diagnostics equal the installed BlackJAX's own estimators run on the stored
    chains, reshaped chain-major exactly as the search documents.
    """
    _skip_missing("nuts")
    import jax.numpy as jnp
    from blackjax import diagnostics

    from autofit.non_linear.search.mcmc.blackjax.nuts.search import (
        split_chain_diagnostics,
    )

    internal = cases.internal_from("nuts")
    info = cases.samples_info_for_comparison(cases.samples_from("nuts"))

    chain_major = jnp.asarray(split_chain_diagnostics(internal["positions"]))

    reference = {
        "ess_per_param": diagnostics.effective_sample_size(chain_major),
        "ess_bulk_per_param": diagnostics.ess_bulk(chain_major),
        "ess_tail_per_param": diagnostics.ess_tail(chain_major),
        "rhat_per_param": diagnostics.rhat(chain_major),
    }

    for key, value in reference.items():
        value = np.atleast_1d(np.asarray(value))
        np.testing.assert_allclose(info[key], value, rtol=1e-12, atol=0.0, err_msg=key)

    assert info["ess_min"] == pytest.approx(min(info["ess_per_param"]), rel=1e-15)
    assert info["ess_bulk_min"] == pytest.approx(min(info["ess_bulk_per_param"]), rel=1e-15)
    assert info["ess_tail_min"] == pytest.approx(min(info["ess_tail_per_param"]), rel=1e-15)
    assert info["rhat_max"] == pytest.approx(max(info["rhat_per_param"]), rel=1e-15)


def test_diagnostics_forward_the_backend_outputs(monkeypatch):
    """
    Ownership: with each BlackJAX estimator replaced by a deterministic double, the
    search forwards its output unchanged to the matching ``samples_info`` key and takes
    the documented min / max.
    """
    _skip_missing("nuts")
    from blackjax import diagnostics

    doubles = {
        "effective_sample_size": [11.0, 7.0, 13.0],
        "ess_bulk": [21.0, 17.0, 23.0],
        "ess_tail": [31.0, 37.0, 33.0],
        "rhat": [1.01, 1.07, 1.03],
    }
    for name, value in doubles.items():
        monkeypatch.setattr(
            diagnostics, name, lambda chains, _value=value: np.asarray(_value)
        )

    internal = cases.internal_from("nuts")
    info = cases.search_from("nuts").samples_info_from(internal)

    assert info["ess_per_param"] == doubles["effective_sample_size"]
    assert info["ess_min"] == 7.0
    assert info["ess_bulk_per_param"] == doubles["ess_bulk"]
    assert info["ess_bulk_min"] == 17.0
    assert info["ess_tail_per_param"] == doubles["ess_tail"]
    assert info["ess_tail_min"] == 31.0
    assert info["rhat_per_param"] == doubles["rhat"]
    assert info["rhat_max"] == 1.07
