import numpy as np

# TODO: Use autofit class?
from scipy import stats

from autofit.mock.mock import Gaussian


def _gaussian(x, centre, intensity, sigma):
    return Gaussian(centre=centre, intensity=intensity, sigma=sigma)(x)


_norm = stats.norm(loc=0, scale=1.)


# TODO: use autofit likelihood
def _likelihood(z, y):
    return np.multiply(-0.5, np.square(np.subtract(z, y)))


def make_data(
        gaussian,
        x
):
    model_line = gaussian(xvalues=x)
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, len(x))
    y = model_line + noise
    return y
