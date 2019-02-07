import numpy as np

from autofit.tools import fit_util


class DataFit(object):

    # noinspection PyUnresolvedReferences
    def __init__(self, data, noise_map, mask, model_data):
        """Class to fit data composed of just one piece of data (e.g. not a list).

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
        self.data = data
        self.noise_map = noise_map
        self.mask = mask
        self.model_data = model_data

        self.residual_map = fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask,
                                                                                model_data=model_data)

        self.chi_squared_map = fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask)

        self.chi_squared = fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=self.chi_squared_map,
                                                                              mask=self.mask)
        self.reduced_chi_squared = self.chi_squared / int(np.size(self.mask) - np.sum(self.mask))
        self.noise_normalization = fit_util.noise_normalization_from_noise_map_and_mask(noise_map=self.noise_map,
                                                                                        mask=self.mask)
        self.likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization)

    @property
    def signal_to_noise_map(self):
        """The signal-to-noise_map of the data and noise-map which are fitted."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map


# noinspection PyUnresolvedReferences
class DataFitStack(object):

    def __init__(self, datas, noise_maps, masks, model_datas):
        """Class to fit to a data-set which is a 'stack' of multiple pieces of datas (stored as lists).

        All fitting quantities (datas, model_datas, residual_maps, etc.) are computed as lists, where the index of \
        each list correspond to index of the data in the stack. The terminology throughout the code is that the stack \
        quantities are pluralized (e.g. noise_maps, residual_maps) and stored as lists.

        Parameters
        -----------
        datas : [ndarray]
            The observed data-set that is fitted (I know datas isn't a word, but it helps with our abstraction :P).
        noise_maps : [ndarray]
            The noise_map-maps of the observed data-set.
        masks: [ndarray]
            The masks that are applied to the data-set.
        model_datas : [ndarray]
            The model-data of the data-set that fits it.

        Attributes
        -----------
        residual_maps : [ndarray]
            List of the residual maps of the model's fit (datas - model_datas).
        chi_squared_maps : [ndarray]
            List of the chi-squared maps of the model's fit ((datas - model_datas) / noise_maps ) **2.0
        chi_squareds : [float]
            List of the overall chi-squareds of the model's fit to the data, summed over every data-point.
        reduced_chi_squareds : [float]
            List of the reduced chi-squared of the model's fit to data (chi_squared / number of datas points), summed
            over every data-point.
        noise_normalizations : [float]
            List of the overall norrmalization term of the noise_map-maps, summed over every data-point.
        likelihoods : [float]
            List of the overall likelihood of the model's fit to the data, summed over evey data-point.
        chi_squared : [float]
            The total chi-squared of the fit (sum of chi_squareds)
        reduced_chi_squared : [floa
            The total reduced chi-squared of the fit (sum of reduced_chi_squareds)
        noise_normalization : float
            The total normalization of the noise_map-maps of the fit (sum of noise_normalizations)
        likelihood : float
            The total likelihood of the fit (sum of chi_squareds)
        """
        self.datas = datas
        self.noise_maps = noise_maps
        self.masks = masks
        self.model_datas = model_datas

        # noinspection PyArgumentList
        self.residual_maps = list(map(lambda data, mask, model_data:
                                      fit_util.residual_map_from_data_mask_and_model_data(data=data, mask=mask,
                                                                                          model_data=model_data),
                                      self.datas, self.masks, self.model_datas))

        # noinspection PyArgumentList
        self.chi_squared_maps = list(map(lambda residual_map, noise_map, mask:
                                         fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
                                             residual_map=residual_map,
                                             noise_map=noise_map,
                                             mask=mask),
                                         self.residual_maps, self.noise_maps, self.masks))

        self.chi_squareds = list(map(lambda chi_squared_map, mask:
                                     fit_util.chi_squared_from_chi_squared_map_and_mask(chi_squared_map=chi_squared_map,
                                                                                        mask=mask),
                                     self.chi_squared_maps, self.masks))

        self.reduced_chi_squareds = list(map(lambda mask, chi_squared_term:
                                             chi_squared_term / int(np.size(mask) - np.sum(mask)),
                                             self.masks, self.chi_squareds))

        self.noise_normalizations = list(map(lambda noise_map, mask:
                                             fit_util.noise_normalization_from_noise_map_and_mask(noise_map=noise_map,
                                                                                                  mask=mask, ),
                                             self.noise_maps, self.masks))

        self.likelihoods = list(map(lambda chi_squared_term, noise_term:
                                    fit_util.likelihood_from_chi_squared_and_noise_normalization(
                                        chi_squared=chi_squared_term,
                                        noise_normalization=noise_term),
                                    self.chi_squareds, self.noise_normalizations))

        self.chi_squared = sum(self.chi_squareds)
        self.reduced_chi_squared = sum(self.reduced_chi_squareds)
        self.noise_normalization = sum(self.noise_normalizations)
        self.likelihood = sum(self.likelihoods)