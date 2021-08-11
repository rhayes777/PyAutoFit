import numpy as np
import pytest
from matplotlib import pyplot as plt

import autofit as af
from autofit import graphical as g
from autofit.messages.normal import NormalMessage


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


def test_uniform():
    message = NormalMessage(
        mean=0.5,
        sigma=0.15
    )

    x = np.linspace(
        0, 1, 100
    )

    def _plot(func):
        plt.plot(
            x, func(x)
        )

    _plot(message.ppf)
    _plot(message.cdf)

    plt.plot(
        x, message.cdf(message.ppf(x))
    )

    plt.show()
