import numpy as np
import pytest

import autofit as af
from autofit import graphical as g
from autofit.messages.normal import UniformNormalMessage
from autofit.messages.transform import LinearShiftTransform


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

    # x = np.linspace(*message._support[0])
    #
    # plt.plot(
    #     x, message.pdf(x)
    # )
    # plt.show()


@pytest.fixture(
    name="ShiftedMessage"
)
def make_shifted_message():
    return UniformNormalMessage.transformed(
        LinearShiftTransform,
        clsname=f"ShiftedUniformNormalMessage"
    )


def test_deferred_transform(
        ShiftedMessage
):
    message = ShiftedMessage(
        shift=1,
        scale=2.1,
        mean=0.0,
        sigma=1.0
    )

    assert np.isnan(message.pdf(0.9))
    assert np.isnan(message.pdf(3.2))
    assert message.pdf(1.5) > 0


def test_values_stay_same(
        ShiftedMessage
):
    message_1 = ShiftedMessage(
        shift=1,
        scale=2.1,
        mean=0.0,
        sigma=1.0
    )
    assert message_1._transform.shift == 1

    ShiftedMessage(
        shift=1,
        scale=2.1,
        mean=0.0,
        sigma=1.0
    )
    assert message_1._transform.shift == 1
