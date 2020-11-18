import autofit as af
from autofit.text import samples_text
import pytest

from os import path

from autofit.mock.mock import MockClassx2

text_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files", "samples")


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

    results_at_sigma = samples_text.summary(samples=samples, sigma=3.0)

    assert "one       1.00 (1.00, 1.20)" in results_at_sigma
    assert "two       2.00 (2.00, 2.20)" in results_at_sigma


def test__latex(samples):

    latex_results_at_sigma = samples_text.latex(samples=samples, sigma=3.0)

    assert r"one_label_{\mathrm{a}} = 1.00^{+0.20}_{-0.00} & " in latex_results_at_sigma
    assert r"two_label_{\mathrm{a}} = 2.00^{+0.20}_{-0.00}" in latex_results_at_sigma
