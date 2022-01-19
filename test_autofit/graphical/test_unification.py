from random import random

import dill
import numpy as np
import pytest

import autofit as af
from autofit import graphical as g
from autofit.messages.normal import UniformNormalMessage
from autofit.messages.transform import log_10_transform


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.GaussianPrior(
        mean=1,
        sigma=2
    )


def test():
    mean_field = g.MeanField({

    })
    mean_field.instance_for_arguments({})


def test_retain_id(
        prior
):
    new_message = prior * prior
    assert new_message.id == prior.id


@pytest.fixture(
    name="x"
)
def make_x():
    return np.linspace(
        0, 1, 100
    )


def test_projected_model():
    model = af.Model(
        af.Gaussian,
        centre=af.UniformPrior()
    )
    samples = af.Samples(
        model,
        [
            af.Sample(
                -1.0, -1.0,
                weight=random(),
                kwargs={
                    ("centre",): random(),
                    ("normalization",): random(),
                    ("sigma",): random(),
                }
            )
            for _ in range(100)
        ]
    )
    result = af.Result(
        samples=samples,
        model=model
    )
    projected_model = result.projected_model

    assert projected_model.prior_count == 3
    assert projected_model.centre is not model.centre
    assert projected_model.centre.id == model.centre.id
    assert isinstance(
        projected_model.centre,
        af.UniformPrior
    )


def test_uniform_normal(x):
    message = UniformNormalMessage.shifted(
        shift=1,
        scale=2.1
    )(
        mean=0.0,
        sigma=1.0
    )

    assert message.pdf(0.9) == 0
    assert message.pdf(3.2) == 0
    assert message.pdf(1.5) > 0


@pytest.mark.parametrize(
    "lower_limit, upper_limit, unit_value, physical_value",
    [
        (0.0, 1.0, 0.5, 0.5),
        (0.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 0.0, 0.0),
        (1.0, 2.0, 0.5, 1.5),
        (1.0, 2.0, 1.0, 2.0),
        (1.0, 2.0, 0.0, 1.0),
        (0.0, 2.0, 0.5, 1.0),
        (0.0, 2.0, 1.0, 2.0),
        (0.0, 2.0, 0.0, 0.0),
    ]
)
def test_uniform_prior(
        lower_limit,
        upper_limit,
        unit_value,
        physical_value
):
    assert af.UniformPrior(
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    ).value_for(
        unit_value
    ) == pytest.approx(
        physical_value
    )


def test_uniform_odd_result():
    prior = af.UniformPrior(90.0, 100.0)
    assert prior.value_for(
        0.0
    ) == pytest.approx(90.0)


@pytest.mark.parametrize(
    "lower_limit",
    [
        1, 90
    ]
)
@pytest.mark.parametrize(
    "upper_limit",
    [
        110, 200
    ]
)
@pytest.mark.parametrize(
    "unit",
    [
        0.00001, 0.5, 0.9
    ]
)
def test_log10(
        lower_limit,
        upper_limit,
        unit
):
    prior = af.LogUniformPrior(
        lower_limit=lower_limit,
        upper_limit=upper_limit
    )

    assert 10.0 ** (
            np.log10(lower_limit)
            + unit * (np.log10(upper_limit) - np.log10(lower_limit))
    ) == pytest.approx(
        prior.value_for(
            unit
        ),
        abs=0.001
    )


@pytest.fixture(
    name="uniform_prior"
)
def make_uniform_prior():
    return af.UniformPrior(
        lower_limit=10,
        upper_limit=20,
        id_=1
    )


def test_prior_arithmetic(
        uniform_prior
):
    multiplied = uniform_prior * uniform_prior
    divided = multiplied / uniform_prior

    multiplied_value = multiplied.value_for(0.3)
    divided_value = divided.value_for(0.3)
    uniform_prior_value = uniform_prior.value_for(0.3)

    assert multiplied_value != divided_value
    assert divided_value == uniform_prior_value


def test_pickle_uniform_prior(
        uniform_prior
):
    pickled_prior = dill.loads(
        dill.dumps(uniform_prior)
    )
    assert pickled_prior == uniform_prior
    assert pickled_prior.id == uniform_prior.id


def test_pickle_log_uniform_prior():
    log_uniform_prior = af.LogUniformPrior()
    pickled_prior = dill.loads(
        dill.dumps(log_uniform_prior)
    )
    assert pickled_prior == log_uniform_prior


@pytest.fixture(
    name="LogMessage"
)
def make_log_message():
    return UniformNormalMessage.shifted(
        shift=1,
        scale=2,
    ).transformed(
        log_10_transform
    )


def test_pickle_transformed(
        LogMessage
):
    dill.loads(
        dill.dumps(LogMessage)
    )


def test_pickle_transformed_instantiated(
        LogMessage
):
    instance = LogMessage(
        mean=1,
        sigma=2
    )
    dill.loads(
        dill.dumps(instance)
    )
