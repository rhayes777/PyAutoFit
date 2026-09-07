"""
Census O2: ``import autofit`` must not pay for ``scipy.special``.

``autofit/messages/normal.py`` builds five module-level ``TransformedMessage``
literals (``UniformNormalMessage`` and friends). ``TransformedMessage`` used to
compute its physical-space support eagerly in ``__init__``, and the inverse of
``phi_transform`` is ``scipy.special.ndtri`` — so every process that touched
``autofit`` at all imported ``scipy.special`` (~0.18s) to fill in a value almost
none of them ever read. The support is now a ``functools.cached_property``.

These tests pin the two halves of that: the import stays clean, and the value
(and its pickling) is unchanged.
"""
import pickle
import subprocess
import sys

import numpy as np
import pytest

import autofit as af
from autofit.messages.composed_transform import TransformedMessage
from autofit.mapper.identifier import Identifier
from autofit.messages.normal import (
    Log10NormalMessage,
    Log10UniformNormalMessage,
    LogNormalMessage,
    MultiLogitNormalMessage,
    NormalMessage,
    UniformNormalMessage,
)
from autofit.messages.transform import phi_transform


def _in_fresh_process(module: str) -> bool:
    """Whether ``module`` is in ``sys.modules`` after a bare ``import autofit``."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import autofit, sys; print({module!r} in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() == "True"


def test__importing_autofit_does_not_import_scipy_special():
    assert _in_fresh_process("scipy.special") is False


def test__importing_autofit_does_not_import_scipy_stats():
    assert _in_fresh_process("scipy.stats") is False


def test__module_level_messages_still_import_and_report_their_support():
    assert UniformNormalMessage._support == ((0.0, 1.0),)
    assert Log10UniformNormalMessage._support == ((1.0, 10.0),)
    assert LogNormalMessage._support == ((0.0, np.inf),)
    assert Log10NormalMessage._support == ((0.0, np.inf),)

    (lower, upper), = MultiLogitNormalMessage._support
    assert lower == 0.0
    assert np.isnan(upper)


def test__support_is_only_computed_on_first_access_and_then_cached():
    message = TransformedMessage(NormalMessage(0, 1), phi_transform)

    assert "_support" not in message.__dict__

    support = message._support

    assert message.__dict__["_support"] is support
    assert message._support is support


def test__pickle_round_trip_preserves_the_support():
    eager = TransformedMessage(NormalMessage(0, 1), phi_transform)._support

    # Un-accessed: the cache is not in the pickled state at all.
    fresh = TransformedMessage(NormalMessage(0, 1), phi_transform)
    assert pickle.loads(pickle.dumps(fresh))._support == eager

    # Accessed: the cached value rides along in ``__dict__`` as it always did.
    assert UniformNormalMessage._support == eager
    assert pickle.loads(pickle.dumps(UniformNormalMessage))._support == eager


@pytest.mark.parametrize(
    "prior, support",
    [
        (af.UniformPrior(lower_limit=0.0, upper_limit=1.0), ((0.0, 1.0),)),
        (af.LogUniformPrior(lower_limit=1e-3, upper_limit=1.0), ((1e-3, 1.0),)),
    ],
)
def test__prior_support_and_limits_survive_a_pickle_round_trip(prior, support):
    assert prior._support == support

    loaded = pickle.loads(pickle.dumps(prior))

    assert loaded._support == support
    assert loaded.lower_limit == prior.lower_limit
    assert loaded.upper_limit == prior.upper_limit


@pytest.mark.parametrize(
    "prior, identifier",
    [
        (
            af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
            "6e85def0116bfefb6478e9f707324a7f",
        ),
        (
            af.LogUniformPrior(lower_limit=1e-3, upper_limit=1.0),
            "84cb3563e638e34128f31008e556880d",
        ),
    ],
)
def test__identifiers_are_byte_identical_to_the_eager_implementation(prior, identifier):
    # Measured on ``main`` before the support was made lazy: an identifier names
    # an output directory, so a change here silently orphans every existing run.
    assert str(Identifier(prior)) == identifier


def test__identifier_does_not_depend_on_whether_the_support_was_accessed():
    accessed = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    assert accessed._support

    untouched = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    assert str(Identifier(accessed)) == str(Identifier(untouched))
