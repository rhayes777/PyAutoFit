import numpy as np


"""
The `Gaussian` class in this module is the model components that is fitted to data using a non-linear search. The
inputs of its __init__ constructor are the parameters which can be fitted for.

The log_likelihood_function in the Analysis class receives an instance of this classes where the values of its
parameters have been set up according to the non-linear search. Because instances of the classes are used, this means
their methods (e.g. profile_from_xvalues) can be used in the log likelihood function.
"""


class Gaussian:
    def __init__(
        self,
        centre=0.0,  # <- **PyAutoFit** recognises these constructor arguments
        intensity=0.1,  # <- are the Gaussian`s model parameters.
        sigma=0.01,
    ):
        self.centre = centre
        self.intensity = intensity
        self.sigma = sigma

    def profile_from_xvalues(self, xvalues):
        """
        Calculate the intensity of the light profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        xvalues : np.ndarray
            The x coordinates in the original reference frame of the data.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )
