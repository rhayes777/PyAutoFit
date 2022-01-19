import autofit as af
from os import path
import os
import matplotlib.pyplot as plt
import numpy as np


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):

        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        """
        Determine the log likelihood of a fit of multiple profiles to the dataset.

        Parameters
        ----------
        instance : af.Collection
            The model instances of the profiles.

        Returnsn
        -------
        fit : Fit.log_likelihood
            The log likelihood value indicating how well this model fit the dataset.

        The `instance` that comes into this method is a Collection. It contains instances of every class
        we instantiated it with, where each instance is named following the names given to the Collection,
        which in this example is a `Gaussian` (with name `gaussian) and Exponential (with name `exponential`):
        """

        xvalues = np.arange(self.data.shape[0])

        try:
            model_data = sum(
                line.profile_1d_via_xvalues_from(xvalues=xvalues) for line in instance
            )
        except TypeError:
            model_data = instance.profile_1d_via_xvalues_from(xvalues=xvalues)

        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)

        return log_likelihood

    def visualize(self, paths, instance, during_analysis):

        """
        During a model-fit, the `visualize` method is called throughout the non-linear search. The `instance` passed
        into the visualize method is maximum log likelihood solution obtained by the model-fit so far and it can be
        used to provide on-the-fly images showing how the model-fit is going.
        """

        xvalues = np.arange(self.data.shape[0])

        try:
            model_data = sum(
                line.profile_1d_via_xvalues_from(xvalues=xvalues) for line in instance
            )
        except TypeError:
            model_data = instance.profile_1d_via_xvalues_from(xvalues=xvalues)

        plt.errorbar(
            x=xvalues,
            y=self.data,
            yerr=self.noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(range(self.data.shape[0]), model_data, color="r")
        plt.title("Dynesty model fit to 1D Gaussian + Exponential dataset.")
        plt.xlabel("x values of profile")
        plt.ylabel("Profile normalization")

        os.makedirs(paths.image_path, exist_ok=True)
        plt.savefig(path.join(paths.image_path, "model_fit.png"))
        plt.clf()
