import numpy as np

"""The 'dataset.py' module has been extended to give the _Dataset_ a name and metadata."""


class Dataset:
    def __init__(self, data, noise_map, name=None):
        """A class containing the data and noise-map of a 1D line _Dataset_.

        Parameters
        ----------
        data : np.ndarray
            The array of the data, in arbitrary units.
        noise_map : np.ndarray
            An array describing the RMS standard deviation error in each data pixel, in arbitrary units.
        """
        self.data = data
        self.noise_map = noise_map

        # The name of the _Dataset_ is used by the aggregator, to determine the name of the file the _Dataset_ is saved as
        # and so that when using the aggregator you can know which _Dataset_ you are manipulating.

        self.name = name if name is str else "dataset"

    @property
    def xvalues(self):
        return np.arange(self.data.shape[0])

    def with_left_trimmed(self, data_trim_left):

        """Here, we use the existing _MaskedDataset_ to create a trimmed _Dataset_."""

        data_trimmed = self.data[data_trim_left:]
        noise_map_trimmed = self.noise_map[data_trim_left:]

        return Dataset(data=data_trimmed, noise_map=noise_map_trimmed)

    def with_right_trimmed(self, data_trim_right):

        """We do the same as above, but removing data to the right."""

        data_trimmed = self.data[:-data_trim_right]
        noise_map_trimmed = self.noise_map[:-data_trim_right]

        return Dataset(data=data_trimmed, noise_map=noise_map_trimmed)


class SettingsMaskedDataset:
    def __init__(self, data_trim_left=None, data_trim_right=None):
        """
        The settings of the _MaskedDataset_ class, that in a phase are used to deterimne if the _MaskedDataset_ is
        trimmed from the left and / or right before model-fitting.

        This class includes tags which are used to customize the output folders of a run dependent on the settings.

        Parameters
        ----------
        data_trim_left : int or None
            The number of pixels in 1D from the left (NumPy index 0) that the _Dataset_ is trimmed.
        data_trim_right : int or None
            The number of pixels in 1D from the right (NumPy index -1) that the _Dataset_ is trimmed.
        """

        self.data_trim_left = data_trim_left
        self.data_trim_right = data_trim_right

    @property
    def tag(self):
        """Generate a tag describin all settings customizing the _MaskedDataset_, which for this example only describes
        how the dataset it trimmed from the left and right.
        """
        return f"{self.data_trim_left_tag}{self.data_trim_right_tag}"

    @property
    def data_trim_left_tag(self):
        """Generate a data trim left tag, to customize phase names based on how much of the _Dataset_ is trimmed to
        its left.

        This changes the phase name 'settings' as follows:

        data_trim_left = None -> settings
        data_trim_left = 2 -> settings__trim_left_2
        data_trim_left = 10 -> settings__trim_left_10
        """
        if self.data_trim_left is None:
            return ""
        return f"__trim_left_{str(self.data_trim_left)}"

    @property
    def data_trim_right_tag(self):
        """Generate a data trim right tag, to customize phase names based on how much of the _Dataset_ is trimmed to its right.

        This changes the phase name 'settings' as follows:

        data_trim_right = None -> settings
        data_trim_right = 2 -> settings__trim_right_2
        data_trim_right = 10 -> settings__trim_right_10
        """
        if self.data_trim_right is None:
            return ""
        return f"__trim_right_{str(self.data_trim_right)}"


class MaskedDataset:
    def __init__(self, dataset, mask, settings=SettingsMaskedDataset()):
        """
        A masked dataset, which is an image, noise-map and mask.

        Parameters
        ----------
        dataset: im.Dataset
            The _Dataset_ (the image, noise-map, etc.)
        mask: msk.Mask
            The 1D mask that is applied to the _Dataset_.
        """

        if settings.data_trim_left is not None:
            dataset = dataset.with_left_trimmed(data_trim_left=settings.data_trim_left)
            mask = mask[settings.data_trim_left :]

        if settings.data_trim_right is not None:
            dataset = dataset.with_right_trimmed(
                data_trim_right=settings.data_trim_right
            )
            mask = mask[: -settings.data_trim_right]

        self.dataset = dataset
        self.mask = mask
        self.data = dataset.data * np.invert(mask)
        self.noise_map = dataset.noise_map * np.invert(mask)

    @property
    def xvalues(self):
        return np.arange(self.data.shape[0])

    def signal_to_noise_map(self):
        return self.data / self.noise_map
