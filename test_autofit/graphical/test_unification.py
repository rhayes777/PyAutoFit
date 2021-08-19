import numpy as np
import pytest

import autofit as af
from autofit import graphical as g
from autofit.mapper.prior.prior import ShiftedMessage
from autofit.messages.normal import UniformNormalMessage


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


def test_bad_id(
        prior
):
    new_message = prior * prior
    new_message.id = 2

    with pytest.raises(
            AssertionError
    ):
        new_message * prior


@pytest.fixture(
    name="x"
)
def make_x():
    return np.linspace(
        0, 1, 100
    )


def test_uniform_normal(x):
    message = UniformNormalMessage.shifted(
        shift=1,
        scale=2.1
    )(
        mean=0.0,
        sigma=1.0
    )

    assert np.isnan(message.pdf(0.9))
    assert np.isnan(message.pdf(3.2))
    assert message.pdf(1.5) > 0


def test_deferred_transform():
    message = ShiftedMessage(
        shift=1,
        scale=2.1,
        mean=0.0,
        sigma=1.0
    )

    assert np.isnan(message.pdf(0.9))
    assert np.isnan(message.pdf(3.2))
    assert message.pdf(1.5) > 0


@pytest.fixture(
    name="message_1"
)
def make_message_1():
    return ShiftedMessage(
        shift=1,
        scale=2.0,
        mean=0.0,
        sigma=1.0
    )


def test_values_stay_same(
        message_1,
):
    assert message_1._transform.shift == 1.0
    assert message_1._transform.scale == 2.0

    message_2 = ShiftedMessage(
        shift=2.0,
        scale=3.0,
        mean=0.0,
        sigma=1.0
    )
    assert message_1._transform.shift == 1
    assert message_1._transform.scale == 2.0

    assert message_2._transform.shift == 2.0
    assert message_2._transform.scale == 3.0


@pytest.mark.parametrize(
    "unit_value, physical_value",
    [
        (0.5, 2),
        (0.0, 1),
        (1.0, 3),
    ]
)
def test_value_for(
        message_1,
        unit_value,
        physical_value
):
    assert message_1.value_for(
        unit_value
    ) == pytest.approx(
        physical_value
    )


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
