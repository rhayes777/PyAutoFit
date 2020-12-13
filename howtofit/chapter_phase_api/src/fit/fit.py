import numpy as np

"""
This module handles a fit, offering a class that computes all relevant quantities of a fit described in chapter 1,
such as the residuals, chi-squared and log_likelihood.
"""


class FitDataset:

    # noinspection PyUnresolvedReferences
    def __init__(self, dataset, model_data):
        """Class to fit a `Dataset` with model data.

        Parameters
        -----------
        dataset : Dataset
            The observed `Dataset` that is fitted.
        model_data : np.ndarray
            The model data the data is fitted with.

        Attributes
        -----------
        residual_map : np.ndarray
            The residual-map of the fit (data - model_data).
        chi_squared_map : np.ndarray
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
    accessible to an instance of to fit class. It is used in the `plot.py` module.
    """

    @property
    def xvalues(self) -> np.ndarray:
        return self.dataset.xvalues

    """Lets use properties to make the `data` and noise-map accessible via our fit."""

    @property
    def data(self) -> np.ndarray:
        return self.dataset.data

    @property
    def noise_map(self) -> np.ndarray:
        return self.dataset.noise_map

    @property
    def residual_map(self) -> np.ndarray:
        """
        Returns the residual-map between the masked dataset and model data, where:

        Residuals = (Data - Model_Data).
        """
        return residual_map_from(data=self.data, model_data=self.model_data)

    @property
    def normalized_residual_map(self) -> np.ndarray:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        return normalized_residual_map_from(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def chi_squared_map(self) -> np.ndarray:
        """
        Returns the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """
        return chi_squared_map_from(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def signal_to_noise_map(self) -> np.ndarray:
        """The signal-to-noise_map of the `Dataset` and noise-map which are fitted."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return chi_squared_from(chi_squared_map=self.chi_squared_map)

    @property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """
        return noise_normalization_from(noise_map=self.noise_map)

    @property
    def log_likelihood(self) -> float:
        """
        Returns the log likelihood of each model data point's fit to the dataset, where:

        Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)
        """
        return log_likelihood_from(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization
        )


def residual_map_from(*, data: np.ndarray, model_data: np.ndarray) -> np.ndarray:
    """
    Returns the residual-map of the fit of model-data to a masked dataset, where:

    Residuals = (Data - Model_Data).

    Parameters
    -----------
    data : np.ndarray
        The data that is fitted.
    mask : np.ndarray
        The mask applied to the dataset, where `False` entries are included in the calculation.
    model_data : np.ndarray
        The model data used to fit the data.
    """
    return np.subtract(data, model_data, out=np.zeros_like(data))


def normalized_residual_map_from(
    *, residual_map: np.ndarray, noise_map: np.ndarray
) -> np.ndarray:
    """
    Returns the normalized residual-map of the fit of model-data to a masked dataset, where:

    Normalized_Residual = (Data - Model_Data) / Noise

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the dataset.
    noise_map : np.ndarray
        The noise-map of the dataset.
    mask : np.ndarray
        The mask applied to the residual-map, where `False` entries are included in the calculation.
    """
    return np.divide(residual_map, noise_map, out=np.zeros_like(residual_map))


def chi_squared_map_from(
    *, residual_map: np.ndarray, noise_map: np.ndarray
) -> np.ndarray:
    """
    Returns the chi-squared-map of the fit of model-data to a masked dataset, where:

    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the dataset.
    noise_map : np.ndarray
        The noise-map of the dataset.
    """
    return np.square(
        np.divide(residual_map, noise_map, out=np.zeros_like(residual_map))
    )


def chi_squared_from(*, chi_squared_map: np.ndarray) -> float:
    """
    Returns the chi-squared terms of a model data's fit to an dataset, by summing the chi-squared-map.

    Parameters
    ----------
    chi_squared_map : np.ndarray
        The chi-squared-map of values of the model-simulator fit to the dataset.
    """
    return float(np.sum(chi_squared_map))


def noise_normalization_from(*, noise_map: np.ndarray) -> float:
    """
    Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map : np.ndarray
        The masked noise-map of the dataset.
    """
    return float(np.sum(np.log(2 * np.pi * noise_map ** 2.0)))


def log_likelihood_from(*, chi_squared: float, noise_normalization: float) -> float:
    """
    Returns the log likelihood of each model data point's fit to the dataset, where:

    Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared : float
        The chi-squared term for the model-simulator fit to the dataset.
    noise_normalization : float
        The normalization noise_map-term for the dataset's noise-map.
    """
    return float(-0.5 * (chi_squared + noise_normalization))
