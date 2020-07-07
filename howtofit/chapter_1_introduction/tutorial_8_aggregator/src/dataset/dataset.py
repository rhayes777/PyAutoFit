import numpy as np

"""The 'dataset.py' module has been extended to give the dataset a name and metadata."""


class Dataset:
    def __init__(self, data, noise_map, name=None):
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

        # The name of the dataset is used by the aggregator, to determine the name of the file the dataset is saved as
        # and so that when using the aggregator you can know which dataset you are manipulating.

        self.name = name if name is str else "dataset"

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

        self.dataset = dataset
        self.mask = mask
        self.data = dataset.data * np.invert(mask)
        self.noise_map = dataset.noise_map * np.invert(mask)

    @property
    def xvalues(self):
        return np.arange(self.data.shape[0])

    def signal_to_noise_map(self):
        return self.data / self.noise_map

    def with_left_trimmed(self, data_trim_left):

        if data_trim_left is None:
            return self

        """Here, we use the existing masked dataset to create a trimmed dataset."""

        data_trimmed = self.dataset.data[data_trim_left:]
        noise_map_trimmed = self.dataset.noise_map[data_trim_left:]

        dataset_trimmed = Dataset(data=data_trimmed, noise_map=noise_map_trimmed)

        mask_trimmed = self.mask[data_trim_left:]

        return MaskedDataset(dataset=dataset_trimmed, mask=mask_trimmed)

    def with_right_trimmed(self, data_trim_right):

        if data_trim_right is None:
            return self

        """We do the same as above, but removing data to the right."""

        data_trimmed = self.dataset.data[:-data_trim_right]
        noise_map_trimmed = self.dataset.noise_map[:-data_trim_right]

        dataset_trimmed = Dataset(data=data_trimmed, noise_map=noise_map_trimmed)

        mask_trimmed = self.mask[:-data_trim_right]

        return MaskedDataset(dataset=dataset_trimmed, mask=mask_trimmed)
