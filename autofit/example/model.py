import math
import numpy as np
from typing import Dict

from autoconf.dictable import Dictable

"""
The `Gaussian` class in this module is the model components that is fitted to data using a non-linear search. The
inputs of its __init__ constructor are the parameters which can be fitted for.

The log_likelihood_function in the Analysis class receives an instance of this classes where the values of its
parameters have been set up according to the non-linear search. Because instances of the classes are used, this means
their methods (e.g. profile_1d_via_xvalues_from) can be used in the log likelihood function.
"""


class Gaussian(Dictable):
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments
        normalization=0.1,  # <- are the Gaussian`s model parameters.
        sigma=0.01,
    ):
        """Represents a 1D `Gaussian` profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a non-linear search.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the `Gaussian` profile.
        sigma
            The sigma value controlling the size of the Gaussian.
        """
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def profile_1d_via_xvalues_from(self, xvalues):
        """
        Calculate the normalization of the profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues : np.ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = xvalues - self.centre

        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )

    def __call__(self, xvalues):
        """
        Calculate the normalization of the profile on a line of Cartesian x coordinates.
        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.
        Parameters
        ----------
        xvalues : np.ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )

    def dict(self) -> Dict:
        """
        Returns the `Gaussian` as a dictionary which can be straight forwardly written to a `.json` file via the
        code:

        with open(model_file, "w+") as f:
            json.dump(gaussian.dict(), f, indent=4)

        Returns
        -------
        The `Gaussian` type and model parameters as a dictionary.

        """
        return super().dict()

    def inverse(
            self,
            y
    ):

        a = self.normalization / (
                y * self.sigma * math.sqrt(2 * math.pi)
        )

        b = 2 * math.log(
            a
        )
        
        return self.centre + self.sigma * math.sqrt(
            b
        )


class Exponential(Dictable):
    def __init__(
        self,
        centre=0.0,  # <- PyAutoFit recognises these constructor arguments are the model
        normalization=0.1,  # <- parameters of the Gaussian.
        rate=0.01,
    ):
        """Represents a 1D Exponential profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a `NonLinearSearch`.

        Parameters
        ----------
        centre
            The x coordinate of the profile centre.
        normalization
            Overall normalization normalisation of the `Gaussian` profile.
        rate
            The decay rate controlling has fast the Exponential declines.
        """
        self.centre = centre
        self.normalization = normalization
        self.rate = rate

    def profile_1d_via_xvalues_from(self, xvalues):
        """
        Calculate the 1D Gaussian profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Exponential, using its centre.

        Parameters
        ----------
        values : np.ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return self.normalization * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )

    def __call__(self, xvalues):
        """
        Calculate the 1D Gaussian profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Exponential, using its centre.

        Parameters
        ----------
        values : np.ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return self.normalization * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )

    def dict(self) -> Dict:
        """
        Returns the `Gaussian` as a dictionary which can be straight forwardly written to a `.json` file via the
        code:

        with open(model_file, "w+") as f:
            json.dump(gaussian.dict(), f, indent=4)

        Returns
        -------
        The `Gaussian` type and model parameters as a dictionary.

        """
        return super().dict()