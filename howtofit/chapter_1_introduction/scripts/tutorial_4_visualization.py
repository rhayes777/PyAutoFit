# %%
"""
Tutorial 4: Visualization
=========================

In this tutorial, we'll extend the `Analysis` class to perform visualization, whereby images showing the model-fits are
output on-the-fly during the `NonLinearSearch`.
"""

# %%
#%matplotlib inline

from pyprojroot import here

workspace_path = str(here())
%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt

# %%
"""
Load the dataset from the `autofit_workspace/dataset` folder.
"""

# %%
dataset_path = path.join("dataset", "howtofit", "chapter_1", "gaussian_x1")
data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

# %%
"""
To perform visualization we'll plot the 1D data as a line. 

To facilitate this we define the function `plot_line` below, which uses Matplotlib to create the 1D plots we've seen 
in previous tutorials. This function has additional inputs so the plot can be output to a specified output path with a 
given output file name.
"""

# %%
def plot_line(
    xvalues,
    line,
    title=None,
    ylabel=None,
    errors=None,
    color="k",
    output_path=None,
    output_filename=None,
):
    """
    Plot a 1D line of data on a plot of x versus y, where the x-axis is the x coordinate of the line and the y-axis
    is the intensity of the line at that coordinate.

    The function include options to output the image to the hard-disk as a .png.

    Parameters
    ----------
    xvalues : np.ndarray
        The x-coordinates the profile is defined on.
    line : np.ndarray
        The intensity values of the profile which are plotted.
    ylabel : str
        The y-label of the plot.
    output_path : str
        The path the image is to be output to hard-disk as a .png.
    output_filename : str
        The filename of the file if it is output as a .png.
    output_format : str
        Determines where the plot is displayed on your screen ("show") or output to the hard-disk as a png ("png").
    """

    plt.errorbar(
        x=xvalues, y=line, yerr=errors, color=color, ecolor="k", elinewidth=1, capsize=2
    )
    plt.title(title)
    plt.xlabel("x value of profile")
    plt.ylabel(ylabel)
    if not path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(path.join(output_path, f"{output_filename}.png"))
    plt.clf()


# %%
"""
In the previous tutorial, we created an `Analysis` class, which defined how our `NonLinearSearch` computed a 
`log_likelihood` to perform the model-fit.

To perform on-the-fly visualization, we simply extend the `Analysis` class with a new method, `visualize`, where
visualization is performed using the `plot_line` function above.
"""

# %%
class Analysis(af.Analysis):
    def __init__(self, data, noise_map):

        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):

        """
        The `log_likelihood_function` is identical to the previous tutorial.
        """

        xvalues = np.arange(self.data.shape[0])

        model_data = instance.profile_from_xvalues(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def visualize(self, paths, instance, during_analysis):

        """
        During a phase, the `visualize` method is called throughout the non-linear search. The `instance` passed into
        the visualize method is highest log likelihood solution obtained by the model-fit so far and it is used
        to output the images defined by the methods below.
        """

        xvalues = np.arange(self.data.shape[0])

        model_data = instance.profile_from_xvalues(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0

        """The visualizer now outputs images of the best-fit results to hard-disk (checkout `visualizer.py`)."""

        plot_line(
            xvalues=xvalues,
            line=self.data,
            title="Data",
            ylabel="Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="data",
        )

        plot_line(
            xvalues=xvalues,
            line=model_data,
            title="Model Data",
            ylabel="Model Data Values",
            color="k",
            output_path=paths.image_path,
            output_filename="model_data",
        )

        plot_line(
            xvalues=xvalues,
            line=residual_map,
            title="Residual Map",
            ylabel="Residuals",
            color="k",
            output_path=paths.image_path,
            output_filename="residual_map",
        )

        plot_line(
            xvalues=xvalues,
            line=chi_squared_map,
            title="Chi-Squared Map",
            ylabel="Chi-Squareds",
            color="k",
            output_path=paths.image_path,
            output_filename="chi_squared_map",
        )


# %%
"""
Lets now repeat the fit of the previous tutorial, but with visualization.
"""

# %%
import gaussian as g

model = af.PriorModel(g.Gaussian)

emcee = af.Emcee(name="tutorial_4", path_prefix=path.join("howtofit", "chapter_1"))

analysis = Analysis(data=data, noise_map=noise_map)

result = emcee.fit(model=model, analysis=analysis)

print(
    "Emcee has begun running - checkout the autofit_workspace/output/howtofit/chapter_1/phase_t4"
    " folder for live output of the results."
    "This Jupyter notebook cell with progress once Emcee has completed - this could take a few minutes!"
)

print("Emcee has finished run - you may now continue the notebook.")

# %%
"""
Lets check that this phase performs visualization. Navigate to the folder `image` in the directory 
`autofit_workspace/output/howtofit/chapter_1. You should see `.png` files containing images of the data, 
residuals, chi-squared map, etc.

Visualization happens `on-the-fly`, such that during `Emcee` these images are output using the current maximum 
likelihood model `Emcee` has found. For models more complex than our 1D `Gaussian` this is useful, as it means we can 
check that `Emcee` has found reasonable solutions during a run and can thus cancel it early if it has ended up with an
incorrect solution.

How often does **PyAutoFit** output new images? This is set by `visualize_every_update` in the config file
`config/visualize/general.ini`.

And with that, we have completed this tutorial.
"""
