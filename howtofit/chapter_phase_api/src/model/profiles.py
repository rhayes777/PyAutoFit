import numpy as np

"""
This module contains the `Gaussian` and `Exponential` classes and model-components we used in chapter 1.

If you are not familiar with Python classes, in particular inheritance and the `super` method below, you may
be unsure what the classes are doing below. I have included comments describing what these command do. For many model
fitting projects many of the models one might fit share common parameters and this inheritance scheme minimizes the
code required to define them.

The Profile class is a base class from which all profiles we add (e.g Gaussian, Exponential, additional profiles
added down the line) will inherit. This is useful, as it signifinies which aspects of our model are different ways of
representing the same thing.
"""


class Profile:
    def __init__(self, centre: float = 0.0, intensity: float = 0.01):
        """Represents an Abstract 1D profile.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the profile.
        """

        """
        Every profile class we add below (e.g. Gaussian, Exponential) will call this __init__ method of the Profile
        base class. Given that every profile will have a centre and intensity, this means we can set these parameters
        in the Profile class`s init method instead of repeating the two lines of code for every individual profile.
        """

        self.centre = centre
        self.intensity = intensity


"""
The inclusion of (Profile) in the `Gaussian` below instructs Python that the `Gaussian` class is going to inherit from
the Profile class.
"""


class Gaussian(Profile):
    def __init__(
        self,
        centre: float = 0.0,  # <- PyAutoFit recognises these constructor arguments
        intensity: float = 0.1,  # <- are the Gaussian`s model parameters.
        sigma: float = 0.01,
    ):
        """Represents a 1D `Gaussian` profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a `NonLinearSearch`.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the `Gaussian` profile.
        sigma : float
            The sigma value controlling the size of the Gaussian.
        """

        """
        Writing (Profile) above does not mean the `Gaussian` class will call the Profile class`s __init__ method. To
        achieve this we have the call the `super` method following the format below.
        """

        super().__init__(centre=centre, intensity=intensity)

        """
        This super method calls the __init__ method of the Profile class above, which means we do not need
        to write the two lines of code below (which are commented out given they are not necessary).
        """

        # self.centre = centre
        # self.intensity = intensity

        self.sigma = sigma  # We still need to set sigma for the Gaussian, of course.

    def profile_from_xvalues(self, xvalues: np.ndarray):
        """
        Calculate the intensity of the profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Gaussian, using its centre.

        Parameters
        ----------
        values : np.ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return np.multiply(
            np.divide(self.intensity, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
        )


class Exponential(Profile):
    def __init__(
        self,
        centre: float = 0.0,  # <- PyAutoFit recognises these constructor arguments are the model
        intensity: float = 0.1,  # <- parameters of the Gaussian.
        rate: float = 0.01,
    ):
        """Represents a 1D Exponential profile, which may be treated as a model-component of PyAutoFit the
        parameters of which are fitted for by a `NonLinearSearch`.

        Parameters
        ----------
        centre : float
            The x coordinate of the profile centre.
        intensity : float
            Overall intensity normalisation of the `Gaussian` profile.
        ratw : float
            The decay rate controlling has fast the Exponential declines.
        """

        super().__init__(centre=centre, intensity=intensity)

        self.rate = rate

    def profile_from_xvalues(self, xvalues: np.ndarray):
        """
        Calculate the intensity of the profile on a line of Cartesian x coordinates.

        The input xvalues are translated to a coordinate system centred on the Exponential, using its centre.

        Parameters
        ----------
        values : np.ndarray
            The x coordinates in the original reference frame of the grid.
        """
        transformed_xvalues = np.subtract(xvalues, self.centre)
        return self.intensity * np.multiply(
            self.rate, np.exp(-1.0 * self.rate * abs(transformed_xvalues))
        )
