import numpy as np

"""The 'dataset.py' module is unchanged from the previous tutorial."""


class Dataset:
    def __init__(self, data, noise_map):
        """A class containing the data and noise-map of a 1D line dataset.

        Parameters
        ----------
        data : np.ndarray
            The array of the data, in arbitrary units.
        noise_map : np.ndarray
            An array describing the RMS standard deviation error in each data pixel, in arbitrary units.
        """
        self.data = data
        self.noise_map = noise_map

    @property
    def xvalues(self):
        return np.arange(self.data.shape[0])


class MaskedDataset:
    def __init__(self, dataset, mask):
        """
        A masked dataset, which is an image, noise-map and mask.

        Parameters
        ----------
        dataset: im.Dataset
            The dataset (the image, noise-map, etc.)
        mask: msk.Mask
            The 1D mask that is applied to the dataset.
        """

        """We store the unmasked dataset in the masked-dataset, incase we need it for anything."""
        self.dataset = dataset

        self.mask = mask

        """We apply the mask, setting all entries where the mask is True to zero."""
        self.data = dataset.data * np.invert(mask)

        """Same for the noise-map"""
        self.noise_map = dataset.noise_map * np.invert(mask)

    @property
    def xvalues(self):
        return np.arange(self.data.shape[0])

    def signal_to_noise_map(self):
        return self.data / self.noise_map
