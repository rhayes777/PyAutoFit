import numpy as np
from scipy import stats

import autofit as af
from autofit import message_passing as mp

n_observations = 100


def make_data():
    gaussian = Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
    x = np.arange(n_observations)
    model_line = gaussian.profile_from_xvalues(xvalues=x)
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, n_observations)
    y = model_line + noise
    return x, y


# TODO: Use autofit class?
def _gaussian(x, centre, intensity, sigma):
    gaussian = Gaussian(centre=centre, intensity=intensity, sigma=sigma)
    return gaussian.profile_from_xvalues(x)


_norm = stats.norm(loc=0, scale=1.)

prior = af.GaussianPrior(
    mean=0,
    sigma=40
)


# TODO: use autofit likelihood
def _likelihood(z, y):
    return _norm.logpdf(z - y)


def test_gaussian():
    x, y = make_data()

    observations = mp.Plate(
        name="observations"
    )

    # TODO: Can we derive variables by looking at function argument names?
    x_ = mp.Variable(
        "x", observations
    )
    y_ = mp.Variable(
        "y", observations
    )
    z = mp.Variable(
        "z", observations
    )
    centre = mp.Variable(
        "centre"
    )
    intensity = mp.Variable(
        "intensity"
    )
    sigma = mp.Variable(
        "sigma"
    )

    gaussian = mp.Factor(
        _gaussian
    )(
        x_,
        centre,
        intensity,
        sigma
    ) == z
    likelihood = mp.Factor(
        _likelihood
    )(z, y_)

    # TODO: Can priors look like autofit priors? Could mp objects derive promise functionality from autofit?
    prior_centre = mp.Factor(
        prior
    )(centre)
    prior_intensity = mp.Factor(
        prior
    )(intensity)
    prior_sigma = mp.Factor(
        prior
    )(sigma)

    model = likelihood * gaussian * prior_centre * prior_sigma * prior_intensity

    model_approx = mp.MeanFieldApproximation.from_kws(
        model,
        centre=mp.NormalMessage.from_prior(
            prior
        ),
        intensity=mp.NormalMessage.from_prior(
            prior
        ),
        sigma=mp.NormalMessage.from_prior(
            prior
        ),
        x=mp.FixedMessage(x),
        y=mp.FixedMessage(y),
        z=mp.NormalMessage.from_mode(
            np.zeros(n_observations), 100
        ),
    )

    opt = mp.optimise.LaplaceOptimiser(
        model_approx,
        n_iter=3
    )
    opt.run()

    for string in ("centre", "intensity", "sigma"):
        print(f"{string} = {opt.model_approx[string].mu}")


class Profile:
    def __init__(self, centre=0.0, intensity=0.01):
        """Represents an Abstract 1D profile.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the profile.
        """
        self.centre = centre
        self.intensity = intensity


class Gaussian(Profile):
    def __init__(
            self,
            centre=0.0,  # <- PyAutoFit recognises these constructor arguments
            intensity=0.1,  # <- are the Gaussian's model parameters.
            sigma=0.01,
    ):
        """Represents a 1D Gaussian profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the Gaussian profile.
        sigma : float
            The sigma value controlling the size of the Gaussian.
        """
        super().__init__(centre=centre, intensity=intensity)
        self.sigma = sigma  # We still need to set sigma for the Gaussian, of course.

    def profile_from_xvalues(self, xvalues):
        """
        Calculate the intensity of the profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        values : ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )
