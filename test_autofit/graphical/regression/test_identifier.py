import pytest

import autofit as af
from autofit.mapper.identifier import Identifier


@pytest.fixture(
    name="truncated_gaussian_prior"
)
def make_truncated_gaussian_prior():
    return Identifier(
        af.TruncatedGaussianPrior(
            mean=1.0,
            sigma=2.0,
            lower_limit="-inf",
            upper_limit="inf"
        )
    )


def test_truncated_gaussian_prior_fields(
        truncated_gaussian_prior
):

    assert truncated_gaussian_prior.hash_list == [
        'TruncatedGaussianPrior',
        'mean',
        '1.0',
        'sigma',
        '2.0',
        'lower_limit',
        '-inf',
        'upper_limit',
        'inf'
    ]


def test_truncated_gaussian_prior(
        truncated_gaussian_prior
):
    assert str(
        truncated_gaussian_prior
    ) == "9a49114940e683d133b12a6d182c85b3"


@pytest.fixture(
    name="uniform_prior"
)
def make_uniform_prior():
    return Identifier(
        af.UniformPrior(
            lower_limit=1.0,
            upper_limit=2.0
        )
    )


def test_uniform_prior_fields(
        uniform_prior
):
    assert uniform_prior.hash_list == [
        'UniformPrior',
        'lower_limit',
        '1.0',
        'upper_limit',
        '2.0'
    ]


def test_uniform_prior(
        uniform_prior
):
    assert str(
        uniform_prior
    ) == "a0de90b9099d70b945dc56094eb5c8de"


@pytest.fixture(
    name="logarithmic_prior"
)
def make_logarithmic_prior():
    return Identifier(
        af.LogUniformPrior(
            lower_limit=1.0,
            upper_limit=2.0
        )
    )


def test_logarithmic_prior_fields(
        logarithmic_prior
):
    assert logarithmic_prior.hash_list == [
        'LogUniformPrior',
        'lower_limit',
        '1.0',
        'upper_limit',
        '2.0'
    ]


def test_logarithmic_prior(
        logarithmic_prior
):
    assert str(
        logarithmic_prior
    ) == "0e8220c88678dcb31a398f9a34dcbc8a"


@pytest.fixture(
    name="model"
)
def make_model():
    return Identifier(
        af.Model(
            af.ex.Gaussian
        )
    )


def test_model_identifier(
        model
):

    assert str(model) == "9929b2be4248f0d116f5c1c034bda870"


def test_model_identifier_fields(
        model
):
    assert model.hash_list == [
        'Model',
        'cls',
        'autofit.example.model.Gaussian',
        'centre',
        'UniformPrior',
        'lower_limit',
        '0.0',
        'upper_limit',
        '1.0',
        'normalization',
        'UniformPrior',
        'lower_limit',
        '0.0',
        'upper_limit',
        '1.0',
        'sigma',
        'UniformPrior',
        'lower_limit',
        '0.0',
        'upper_limit',
        '1.0'
    ]
