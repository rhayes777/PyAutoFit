import numpy as np

"""
This is our dataset, which in tutorial 1 stores a 1D Gaussian with a noise-map.

Our data is a 1D numpy array of values corresponding to the observed counts of the Gaussian.
The noise-map corresponds to the expected noise in every data point.

In PyAutoFit we use the term 'dataset' to represent the collection of data, noise-map, etc. that are fitted. In this
chapter our dataset is a 1D line, for your model-fitting problem your dataset is probably more complex than this.

In these tutorials we are calling our dataset class 'Dataset', as opposed to something more specific to the data
(e.g. 'Line'). We advise that the dataset class is give a more specific name (e.g., if your dataset is imaging data,
the class may be called 'Imaging'. If your dataset is a spectrum, you may use 'Spectrum').

However, to make the tutorials clear, we've stuck with the name 'Dataset' for clarity. The example project at the
end of the chapter will adopt a more specific dataset naming convention.
"""


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

    """The 'xvalues' defines the x coordinates of the 1D Gaussian, which we assume run from 0 the length of the data."""

    @property
    def xvalues(self):
        return np.arange(self.data.shape[0])
