import numpy as np

"""
This module handles a fit, offering a class that computes all relevent quantities of a fit described in tutorial 2,
such as the residuals, chi-squared and log_likelihood.
"""


class FitDataset:

    # noinspection PyUnresolvedReferences
    def __init__(self, dataset, model_data):
        """Class to fit a dataset with model data.

        Parameters
        -----------
        dataset : ndarray
            The observed dataset that is fitted.
        model_data : ndarray
            The model data the data is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual-map of the fit (data - model_data).
        chi_squared_map : ndarray
            The chi-squared-map of the fit ((data - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every data point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of datas points), summed over \
            every data point.
        noise_normalization : float
            The overall normalization term of the noise_map, summed over every data point.
        log_likelihood : float
            The overall log likelihood of the model's fit to the dataset, summed over evey data point.
        """

        self.dataset = dataset
        self.model_data = model_data

    """
    This is a convenience method that makes the dataset's xvalues (used to generate the model data) directly
    accessible to an instance of to fit class. It is used in the 'plot.py' module.
    """

    @property
    def xvalues(self):
        return self.dataset.xvalues

    """Lets use properties to make the 'data' and noise-map accessible via our fit."""

    @property
    def data(self):
        return self.dataset.data

    @property
    def noise_map(self):
        return self.dataset.noise_map

    @property
    def residual_map(self):
        return residual_map_from_data_and_model_data(
            data=self.data, model_data=self.model_data
        )

    @property
    def normalized_residual_map(self):
        return normalized_residual_map_from_residual_map_and_noise_map(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def chi_squared_map(self):
        return chi_squared_map_from_residual_map_and_noise_map(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def signal_to_noise_map(self):
        """The signal-to-noise_map of the dataset and noise-map which are fitted."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def chi_squared(self):
        return chi_squared_from_chi_squared_map(chi_squared_map=self.chi_squared_map)

    @property
    def noise_normalization(self):
        return noise_normalization_from_noise_map(noise_map=self.noise_map)

    @property
    def log_likelihood(self):
        return likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization
        )


def residual_map_from_data_and_model_data(data, model_data):
    """Compute the residual-map between a masked observed data and model-data, where:

    Residuals = (Data - Model_Data).

    Parameters
    -----------
    data : np.ndarray
        The observed data that is fitted.
    mask : np.ndarray
        The mask applied to the dataset, where *False* entries are included in the calculation.
    model_data : np.ndarray
        The model-data used to fit the observed data.
    """
    return np.subtract(data, model_data, out=np.zeros_like(data))


def normalized_residual_map_from_residual_map_and_noise_map(residual_map, noise_map):
    """Compute the normalized residual-map between a masked observed data and model-data, where:

    Normalized_Residual = (Data - Model_Data) / Noise

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-data fit to the observed data.
    noise_map : np.ndarray
        The noise-map of the observed dataset.
    """
    return np.divide(residual_map, noise_map, out=np.zeros_like(residual_map))


def chi_squared_map_from_residual_map_and_noise_map(residual_map, noise_map):
    """Computes the chi-squared-map between a residual-map and noise-map, where:

    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-data fit to the observed data.
    noise_map : np.ndarray
        The noise-map of the observed data.
    """
    return np.square(
        np.divide(residual_map, noise_map, out=np.zeros_like(residual_map))
    )


def chi_squared_from_chi_squared_map(chi_squared_map):
    """Compute the chi-squared terms of each model data's fit to an observed dataset, by summing the masked
    chi-squared-map of the fit.

    Parameters
    ----------
    chi_squared_map : np.ndarray
        The chi-squared-map of values of the model-data fit to the observed dataset.
    mask : np.ndarray
        The mask applied to the chi-squared-map, where *False* entries are included in the calculation.
    """
    return np.sum(chi_squared_map)


def noise_normalization_from_noise_map(noise_map):
    """Compute the noise-map normalization terms of a noise-map, summing the value in every pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map : np.ndarray
        The noise-map of the observed dataset.
    """
    return np.sum(np.log(2 * np.pi * noise_map ** 2.0))


def likelihood_from_chi_squared_and_noise_normalization(
    chi_squared, noise_normalization
):
    """Compute the log likelihood of each 1D model-data fit to the dataset, where:

    Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared : float
        The chi-squared term for the model-data fit to the observed dataset.
    noise_normalization : float
        The normalization noise_map-term for the observed dataset's noise-map.
    """
    return -0.5 * (chi_squared + noise_normalization)
