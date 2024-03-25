import itertools

import autofit as af
import pytest

from autofit.text.text_util import result_info_from


@pytest.fixture(autouse=True)
def reset_ids():
    af.Prior._ids = itertools.count()


@pytest.fixture(name="collection")
def make_collection():
    centre = af.UniformPrior(0.0, 1.0)
    normalization = af.UniformPrior(0.0, 1.0)
    sigma = af.UniformPrior(0.0, 1.0)
    return af.Collection(
        [
            af.Model(
                af.Gaussian,
                centre=centre,
                normalization=normalization,
                sigma=sigma,
            )
            for _ in range(20)
        ]
    )


def test_model_info(collection):
    assert (
        collection.info
        == """Total Free Parameters = 3

model                                                                           Collection (N=3)
    0 - 9                                                                       Gaussian (N=3)

0 - 9
    centre                                                                      UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
    normalization                                                               UniformPrior [1], lower_limit = 0.0, upper_limit = 1.0
    sigma                                                                       UniformPrior [2], lower_limit = 0.0, upper_limit = 1.0"""
    )


@pytest.fixture(name="samples")
def make_samples(collection):
    parameters = [len(collection) * [1.0, 2.0, 3.0]]

    log_likelihood_list = [1.0]

    return af.m.MockSamples(
        model=collection,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0],
            weight_list=log_likelihood_list,
            model=collection,
        ),
        # max_log_likelihood_instance=collection.instance_from_prior_medians(),
    )


def test_model_results(samples):
    assert (
        result_info_from(samples)
        == """Maximum Log Likelihood                                                          1.00000000
Maximum Log Posterior                                                           1.00000000

model                                                                           Collection (N=3)
    0 - 9                                                                       Gaussian (N=3)

Maximum Log Likelihood Model:

19
    centre                                                                      1.000
    normalization                                                               2.000
    sigma                                                                       3.000

 WARNING: The samples have not converged enough to compute a PDF and model errors. 
 The model below over estimates errors. 



Summary (1.0 sigma limits):

19
    centre                                                                      1.00 (1.00, 1.00)
    normalization                                                               2.00 (2.00, 2.00)
    sigma                                                                       3.00 (3.00, 3.00)

instances

"""
    )
