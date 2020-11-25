import numpy as np

"""
This is our dataset, which stores data containing a noisy 1D `Gaussian` with a noise-map.

Our data is a 1D numpy array of values corresponding to the observed counts of the Gaussian. The noise-map corresponds 
to the expected noise in every data point.

In PyAutoFit we use the term `Dataset` to represent the collection of data, noise-map, etc. that are fitted. For your 
model-fitting problem your `Dataset` is most likely more complex than this and could contain many additional 
components than just the `data` and `noise_map`.

In these tutorials we are calling our dataset class `Dataset`, as opposed to something more specific to the data
(e.g. `Line`). We advise that for your project the `Dataset` class is give a more specific name (e.g., if your 
`Dataset` is imaging data, the class may be called `Imaging`. If your `Dataset` is a spectrum, you may use `Spectrum`).

However, to make the tutorials generic, we've stuck with the name `Dataset` for clarity.

NOTE: Many methods in this module (those associated with trimming and Settings) are described in tutorial 3 of the
phase API chapter.
"""

"""
We use a `SettingsDataset` class to choose the settings of how our `Dataset` is setup for the model-fit. For this 
template project, these settings control just one aspect of the `Dataset`, how it is trimmed from the right and left
before model-fitting.

The `Settings` class includes tags, which customize the folders of the output of the phase. See tutorial 3 and the 
module, `settings.py` for a more complete description of tagging.
"""


class SettingsDataset:
    def __init__(self, data_trim_left: int = None, data_trim_right: int = None):
        """
        The settings of the `Dataset` class, that a phase uses to determine if the `Dataset` is trimmed from the left
        and / or right before model-fitting and how many pixels are trimmed.

        This class includes tags which are used to customize the output folders of a run dependent on the settings.

        Parameters
        ----------
        data_trim_left : int or None
            The number of pixels in 1D from the left (NumPy index 0) that the `Dataset` is trimmed.
        data_trim_right : int or None
            The number of pixels in 1D from the right (NumPy index -1) that the `Dataset` is trimmed.
        """

        self.data_trim_left = data_trim_left
        self.data_trim_right = data_trim_right

    @property
    def tag(self) -> str:
        """
        Returns a tag describing all settings customizing the `Dataset`, which for this example only describes
        how the dataset is trimmed from the left and right.
        """
        return f"__dataset[{self.data_trim_left_tag}{self.data_trim_right_tag}]"

    @property
    def data_trim_left_tag(self) -> str:
        """
        Returns a data trim left tag, to customize phase names based on how much of the `Dataset` is trimmed to
        its left.

        This changes the phase name `settings` as follows:

        data_trim_left = None -> settings
        data_trim_left = 2 -> settings__trim_left_2
        data_trim_left = 10 -> settings__trim_left_10
        """
        if self.data_trim_left is None:
            return ""
        return f"trim_left_{self.data_trim_left}"

    @property
    def data_trim_right_tag(self) -> str:
        """
        Returns a data trim right tag, to customize phase names based on how much of the `Dataset` is trimmed to
        its right.

        This changes the phase name `settings` as follows:

        data_trim_right = None -> settings
        data_trim_right = 2 -> settings__trim_right_2
        data_trim_right = 10 -> settings__trim_right_10
        """
        if self.data_trim_right is None:
            return ""
        return f"__trim_right_{self.data_trim_right}"


class Dataset:
    def __init__(self, data: np.ndarray, noise_map: np.ndarray):
        """
        A class containing the data and noise-map of a 1D `Dataset`.

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
    def xvalues(self) -> np.ndarray:
        """
        The x coordinates of `Dataset` used the evaluated the model profiles, which run from 0 to the length of the
        data.
        """
        return np.arange(self.data.shape[0])

    def trimmed_dataset_from_settings(self, settings: SettingsDataset) -> "Dataset":
        """
        Returns a trimmed `Dataset` based on an input `SettingsDataset`.

        Parameters
        ----------
        settings : SettingsDataset
            The dataset settings that define whether the `Dataset` is trimmed and by how much.
        """

        data = self.data
        noise_map = self.noise_map

        if settings.data_trim_left is not None:
            data = data[settings.data_trim_left :]
            noise_map = noise_map[settings.data_trim_left :]

        if settings.data_trim_right is not None:
            data = data[: -settings.data_trim_right]
            noise_map = noise_map[: -settings.data_trim_right]

        return Dataset(data=data, noise_map=noise_map)
