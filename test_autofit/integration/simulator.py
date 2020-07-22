import os
from astropy.io import fits
from test_autofit.integration.src.model import profiles

import numpy as np

# %%
#%matplotlib inline

# %%
"""
___Simulator___
This script simulates the 1D Gaussians line profile datasets used by the integration tests.
"""


def numpy_array_1d_to_fits(array_1d, file_path, overwrite=False):
    """Write a 1D NumPy array to a .fits file.

    Parameters
    ----------
    array_1d : ndarray
        The 1D array that is written to fits.
    file_path : str
        The full path of the file that is output, including the file name and '.fits' extension.
    overwrite : bool
        If True and a file already exists with the input file_path the .fits file is overwritten. If False, an error \
        will be raised.

    Returns
    -------
    None

    Examples
    --------
    array_1d = np.ones(shape=(,5))
    numpy_array_1d_to_fits(array_1d=array_1d, file_path='/path/to/file/filename.fits', overwrite=True)
    """

    if overwrite and os.path.exists(file_path):
        os.remove(file_path)
        
    new_hdr = fits.Header()
    hdu = fits.PrimaryHDU(array_1d, new_hdr)
    hdu.writeto(file_path)


# %%
"""
The path to the chapter and dataset folder on your computer. The data should be distributed with PyAutoFit, however
if you wish to reuse this script to generate it again (or new datasets) you must update the paths appropriately.
"""

# %%
test_path = "{}".format(os.path.dirname(os.path.realpath(__file__)))
dataset_path = f"{test_path}/dataset"

# %%
"""
__Gaussian X1__

Setup the path and filename the .fits file of the Gaussian is written to.
"""

# %%
data_path = f"{dataset_path}/gaussian_x1"

# %%
"""
Create a model instance of the Gaussian.
"""

# %%
gaussian = profiles.Gaussian(centre=50.0, intensity=25.0, sigma=10.0)

# %%
"""
Specify the number of pixels used to create the xvalues on which the 1D line of the profile is generated using and
thus defining the number of data-points in our data.
"""

# %%
pixels = 100
xvalues = np.arange(pixels)

# %%
"""
Evaluate this Gaussian model instance at every xvalues to create its model line profile.
"""

# %%
model_line = gaussian.profile_from_xvalues(xvalues=xvalues)

# %%
"""
Determine the noise (at a specified signal to noise level) in every pixel of our model line profile.
"""

# %%
signal_to_noise_ratio = 25.0
noise = np.random.normal(0.0, 1.0 / signal_to_noise_ratio, pixels)

# %%
"""
Add this noise to the model line to create the line data that is fitted, using the signal-to-noise ratio to compute
noise-map of our data which is required when evaluating the chi-squared value of the likelihood.
"""

# %%
data = model_line + noise
noise_map = (1.0 / signal_to_noise_ratio) * np.ones(pixels)

# %%
"""
Output this data to fits file, so it can be loaded and fitted in the HowToFit tutorials.
"""

# %%
numpy_array_1d_to_fits(
    array_1d=data, file_path=f"{data_path}/data.fits", overwrite=True
)
numpy_array_1d_to_fits(
    array_1d=noise_map, file_path=f"{data_path}/noise_map.fits", overwrite=True
)