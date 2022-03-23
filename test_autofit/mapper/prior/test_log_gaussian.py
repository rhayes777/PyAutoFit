import numpy as np
from matplotlib import pyplot as plt

import autofit as af
from autofit.messages import LogNormalMessage


def test():
    # message = LogNormalMessage(
    #     mean=1.0,
    #     sigma=2.0
    # )
    # print(message)
    # x = np.linspace(
    #     start=0,
    #     stop=100,
    #     num=100,
    # )
    #
    # y = list(map(
    #     message.pdf,
    #     x
    # ))
    #
    # plt.plot(x, y)
    # plt.show()

    log_gaussian = af.LogGaussianPrior(
        mean=1.0,
        sigma=2.0
    )
    assert log_gaussian.value_for(0.0) == 0.0
