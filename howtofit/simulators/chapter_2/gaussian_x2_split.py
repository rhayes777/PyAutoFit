from howtofit.simulators.chapter_1 import profiles

import numpy as np

# %%
"""
__Gaussian x2 Split__

Create the model instances of the two Gaussians.
"""

# %%
gaussian_0 = profiles.Gaussian(centre=25.0, intensity=50.0, sigma=12.5)
gaussian_1 = profiles.Gaussian(centre=75.0, intensity=50.0, sigma=12.5)

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
Evaluate both Gaussian model instances at every xvalues to create their model profiles and sum them
together to create the overall model profile.
"""

# %%
model_line = gaussian_0.profile_from_xvalues(
    xvalues=xvalues
) + gaussian_1.profile_from_xvalues(xvalues=xvalues)

# %%
"""
Determine the noise (at a specified signal to noise level) in every pixel of our model profile.
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
