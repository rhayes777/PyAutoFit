import autofit as af
from autofit.text import samples_text
import pytest

import os

from test_autofit.mock import MockClassx2

text_path = "{}/files/samples/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name="model")
def make_model():
    return af.ModelMapper(mock_class=MockClassx2)


@pytest.fixture(name="samples")
def make_samples(model):
    parameters = [[1.0, 2.0], [1.2, 2.2]]

    log_likelihoods = [1.0, 0.0]

    return af.PDFSamples(
        model=model,
        parameters=parameters,
        log_likelihoods=log_likelihoods,
        log_priors=[],
        weights=log_likelihoods,
    )


def test__summary(samples):

    results_at_sigma = samples_text.summary(
        samples=samples, sigma=3.0
    )

    assert "Median PDF model Summary (3.0 sigma limits):" in results_at_sigma
    assert "one_label 1.00 (1.00, 1.20)" in results_at_sigma
    assert "two_label 2.10 (2.00, 2.20)"

    results_at_sigma = samples_text.summary(
        samples=samples, sigma=3.0, name_to_label=False
    )

    assert "Median PDF model Summary (3.0 sigma limits):" in results_at_sigma
    assert "one 1.00 (1.00, 1.20)" in results_at_sigma
    assert "two 2.10 (2.00, 2.20)"

def test__table(samples):

    results_at_sigma = samples_text.table(
        samples=samples, sigma=3.0
    )

    assert "Median PDF model Table (3.0 sigma limits):" in results_at_sigma
    assert (
        "one_label_a 1.00 (1.00, 1.20) & two_label_a 2.00 (2.00, 2.20)"
        in results_at_sigma
    )


def test__latex(samples):

    latex_results_at_sigma = samples_text.latex(
        samples=samples, sigma=3.0
    )

    assert (
        r"one_label_{\mathrm{a}} = 1.00^{+0.20}_{-0.00} & " in latex_results_at_sigma
    )
    assert (
        r"two_label_{\mathrm{a}} = 2.00^{+0.20}_{-0.00}" in latex_results_at_sigma
    )

