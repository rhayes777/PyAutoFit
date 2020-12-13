# %%
"""
Tutorial 7: Fitting Multiple Datasets
=====================================

In this tutorial, we'll fit multiple dataset's with the same `NonLinearSearch`, producing multiple sets of results on
our hard-disc. In the following tutorials, we then use these results and the `Aggregator` to load the results into
our Jupyter notebook to interpret, inspect and plot the output results.

we'll fit 3 different dataset`s, each with a single `Gaussian` model.
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
import pickle

# %%
"""
We'll reuse the `plot_line` and `Analysis` classes of the previous tutorial.

Note that the `Analysis` class has a new method, `save_attributes_for_aggregator`. This method specifies which properties of the
fit are output to hard-disc so that we can load them using the `Aggregator` in the next tutorial.
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


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        model_data = self.model_data_from_instance(instance=instance)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        chi_squared = sum(chi_squared_map)
        noise_normalization = np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        log_likelihood = -0.5 * (chi_squared + noise_normalization)

        return log_likelihood

    def model_data_from_instance(self, instance):
        """
        To create the summed profile of all individual profiles in an instance, we can use a dictionary comprehension
        to iterate over all profiles in the instance.
        """

        xvalues = np.arange(self.data.shape[0])

        return sum(
            [profile.profile_from_xvalues(xvalues=xvalues) for profile in instance]
        )

    def visualize(self, paths, instance, during_analysis):
        """
        This method is identical to the previous tutorial, except it now uses the `model_data_from_instance` method
        to create the profile.
        """

        xvalues = np.arange(self.data.shape[0])

        model_data = self.model_data_from_instance(instance=instance)

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

    def save_attributes_for_aggregator(self, paths):
        """Save files like the data and noise-map as pickle files so they can be loaded in the `Aggregator`"""

        # These functions save the objects we will later access using the aggregator. They are saved via the `pickle`
        # module in Python, which serializes the data on to the hard-disk.

        with open(path.join(f"{paths.pickle_path}", "data.pickle"), "wb") as f:
            pickle.dump(self.data, f)

        with open(path.join(f"{paths.pickle_path}", "noise_map.pickle"), "wb") as f:
            pickle.dump(self.noise_map, f)


"""We'll also fit the same model as the previous tutorial."""

import profiles as p

model = af.CollectionPriorModel(gaussian=p.Gaussian)

# %%
"""
Here, for each dataset we are going to set up the correct path, load it, and fit it using a `NonLinearSearch`.

We want our results to be in a folder specific to the dataset. we'll use the dataset`'s name string to do this. Lets
create a list of all 3 of our dataset names.
"""

# %%
dataset_names = ["gaussian_x1_0", "gaussian_x1_1", "gaussian_x1_2"]

# %%
"""
We can also attach information to the model-fit, by setting up an info dictionary. 

Information about our model-fit (e.g. the data of observation) that isn't part of the model-fit is made accessible to 
the `Aggregator`. For example, below we write info on the dataset's data of observation and exposure time.
"""

# %%
info = {"date_of_observation": "01-02-18", "exposure_time": 1000.0}

# %%
"""
This for loop runs over every dataset, checkout the comments below for how we set up the path structure.
"""

# %%
for dataset_name in dataset_names:

    """Load the dataset from the `autofit_workspace/dataset` folder."""

    dataset_path = path.join("dataset", "howtofit", "chapter_1", dataset_name)

    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    """
    Here, we create the `Emcee` `NonLinear` as normal. However, we also includethe data name in the `path_prefix`.
    
    Note that we pass the info to the fit, so that the `Aggregator` can make it accessible.
    """

    analysis = Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(
        name="tutorial_7_multi",
        path_prefix=f"howtofit/chapter_1/aggregator/{dataset_name}",
    )

    print(
        f"Emcee has begun running, checkout \n"
        f"autofit_workspace/output/howtofit/chapter_1/aggregator/{dataset_name}/tutorial_7_multi folder for live \n"
        f"output of the results. This Jupyter notebook cell with progress once Emcee has completed, this could take a \n"
        f"few minutes!"
    )

    emcee.fit(model=model, analysis=analysis, info=info)

# %%
"""
Checkout the output folder, you should see three new sets of results corresponding to our 3 `Gaussian` datasets.

Unlike previous tutorials, these folders in the output folder are named after the dataset and contain the folder
with the phase`s name, as opposed to just the phase-name folder.
"""
