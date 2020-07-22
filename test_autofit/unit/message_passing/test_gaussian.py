import numpy as np


def make_data():
    gaussian = Gaussian(centre=50.0, intensity=25.0, sigma=10.0)
    pixels = 100
    xvalues = np.arange(pixels)
    model_line = gaussian.profile_from_xvalues(xvalues=xvalues)
    signal_to_noise_ratio = 25.0
    noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)
    data = model_line + noise
    return data


def test_gaussian():
    data = make_data()
    n_obs, = data.shape


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
