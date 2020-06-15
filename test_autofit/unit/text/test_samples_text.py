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
    parameters = [[1.0, 2.0],
                  [1.2, 2.2],
                  ]

    log_likelihoods = [1.0, 0.0]

    return af.PDFSamples(
        model=model,
        parameters=parameters,
        log_likelihoods=log_likelihoods,
        log_priors=[],
        weights=log_likelihoods
    )


def test__results_at_sigma_from_sigma(samples):
    results_at_sigma = samples_text.results_at_sigma_from_samples(samples=samples, sigma=3.0)

    assert "Most probable model (3.0 sigma limits):" in results_at_sigma
    assert "one                                           " \
           "                                        1.100 (1.000, 1.200)" in results_at_sigma
    assert "two                                      " \
           "                                             2.100 (2.000, 2.200)"


def test__results_latex_from_sigma(samples):
    latex_results_at_sigma = samples_text.latex_results_at_sigma_from_samples(samples=samples, sigma=3.0)

    print(latex_results_at_sigma[0])

    assert latex_results_at_sigma[0] == 'x4p0_{\\mathrm{a}} = 1.10^{+1.20}_{-1.00} & '
    assert latex_results_at_sigma[1] == 'x4p1_{\\mathrm{a}} = 2.10^{+2.20}_{-2.00} & '


def test__results_to_file(samples):
    file_results = f"{text_path}model.results"

    samples_text.results_to_file(samples=samples, file_results=file_results, during_analysis=True)

    results = open(file_results)

    line = results.readline()

    assert line == "Maximum Likelihood                                                                        1.00000000\n"
