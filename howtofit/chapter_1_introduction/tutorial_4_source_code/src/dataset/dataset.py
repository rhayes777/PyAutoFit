import numpy as np

"""
This is our dataset, which in tutorial 1 stores a 1D _Gaussian_ with a noise-map.

Our data is a 1D numpy array of values corresponding to the observed counts of the Gaussian.
The noise-map corresponds to the expected noise in every data point.

In PyAutoFit we use the term 'dataset' to represent the collection of data, noise-map, etc. that are fitted. In this
chapter our _Dataset_ is a 1D line, for your model-fitting problem your _Dataset_ is probably more complex than this.

In these tutorials we are calling our _Dataset_ class 'Dataset', as opposed to something more specific to the data
(e.g. 'Line'). We advise that the _Dataset_ class is give a more specific name (e.g., if your _Dataset_ is imaging data,
the class may be called 'Imaging'. If your _Dataset_ is a spectrum, you may use 'Spectrum').

However, to make the tutorials clear, we've stuck with the name 'Dataset' for clarity. The example project at the
end of the chapter will adopt a more specific _Dataset_ naming convention.
"""


class Dataset:
    def __init__(self, data, noise_map):
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

    """The 'xvalues' defines the x coordinates of the 1D Gaussian, which we assume run from 0 the length of the data."""

    @property
    def xvalues(self):
        return np.arange(self.data.shape[0])
