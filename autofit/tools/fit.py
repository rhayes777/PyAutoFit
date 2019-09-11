import numpy as np

from autofit.tools import fit_util


class DataFit(object):

    # noinspection PyUnresolvedReferences
    def __init__(self, data, noise_map, mask, model_data):
        """Class to fit data where the data arrays are any dimension.

        Parameters
        -----------
        data : ndarray
            The observed data that is fitted.
        noise_map : ndarray
            The noise_map-map of the observed data.
        mask: msk.Mask
            The masks that is applied to the data.
        model_data : ndarray
            The model data the fitting image is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual map of the fit (datas - model_data).
        chi_squared_map : ndarray
            The chi-squared map of the fit ((datas - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the data, summed over every data-point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to data (chi_squared / number of datas points), summed over \
            every data-point.
        noise_normalization : float
            The overall normalization term of the noise_map-map, summed over every data-point.
        likelihood : float
            The overall likelihood of the model's fit to the data, summed over evey data-point.
        """
        self._data = data
        self._noise_map = noise_map
        self._mask = mask
        self._model_data = model_data

        self._residual_map = fit_util.residual_map_from_data_mask_and_model_data(
            data=data, mask=mask, model_data=model_data)

        self._chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=self._residual_map, noise_map=self._noise_map, mask=self._mask)

        self.chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=self._chi_squared_map, mask=self._mask)

        self.reduced_chi_squared = self.chi_squared / int(
            np.size(self._mask) - np.sum(self._mask))

        self.noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(
            noise_map=self._noise_map,
            mask=self._mask)

        self.likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization)

    @property
    def _normalized_residual_map(self):
        return fit_util.normalized_residual_map_from_residual_map_noise_map_and_mask(
            residual_map=self._residual_map, noise_map=self._noise_map, mask=self._mask)

    @property
    def _signal_to_noise_map(self):
        """The signal-to-noise_map of the data and noise-map which are fitted."""
        signal_to_noise_map = np.divide(self._data, self._noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map