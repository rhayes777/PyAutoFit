import json
from os import path
import os
import matplotlib.pyplot as plt
from typing import List

from autofit.jax_wrapper import numpy as np

import autofit as af

"""
The `analysis.py` module contains the dataset and log likelihood function which given a model instance (set up by
the non-linear search) fits the dataset and returns the log likelihood of that model.
"""

class Analysis(af.Analysis):
    def __init__(self, data: np.ndarray, noise_map:np.ndarray):
        """
        In this example the `Analysis` object only contains the data and noise-map. It can be easily extended,
        for more complex data-sets and model fitting problems.

        Parameters
        ----------
        data
            A 1D numpy array containing the data (e.g. a noisy 1D Gaussian) fitted in the workspace examples.
        noise_map
            A 1D numpy array containing the noise values of the data, used for computing the goodness of fit
            metric.
        """
        super().__init__()

        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Determine the log likelihood of a fit of multiple profiles to the dataset.

        Parameters
        ----------
        instance : af.Collection
            The model instances of the profiles.

        Returns
        -------
        The log likelihood value indicating how well this model fit the dataset.
        """

        xvalues = np.arange(self.data.shape[0])
        model_data_1d = np.zeros(self.data.shape[0])

        try:
            for profile in instance:
                try:
                    model_data_1d += profile.model_data_1d_via_xvalues_from(xvalues=xvalues)
                except AttributeError:
                    pass
        except TypeError:
            model_data_1d += instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

        residual_map = self.data - model_data_1d
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)

        return log_likelihood

    def visualize(self, paths: af.DirectoryPaths, instance: af.ModelInstance, during_analysis : bool):
        """
        During a model-fit, the `visualize` method is called throughout the non-linear search and is used to output
        images indicating the quality of the fit so far..

        The `instance` passed into the visualize method is maximum log likelihood solution obtained by the model-fit
        so far and it can be used to provide on-the-fly images showing how the model-fit is going.

        For your model-fitting problem this function will be overwritten with plotting functions specific to your
        problem.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).
        during_analysis
            If True the visualization is being performed midway through the non-linear search before it is finished,
            which may change which images are output.
        """

        xvalues = np.arange(self.data.shape[0])
        model_data_1d = np.zeros(self.data.shape[0])

        try:
            for profile in instance:
                try:
                    model_data_1d += profile.model_data_1d_via_xvalues_from(xvalues=xvalues)
                except AttributeError:
                    pass
        except TypeError:
            model_data_1d += instance.model_data_1d_via_xvalues_from(xvalues=xvalues)

        plt.errorbar(
            x=xvalues,
            y=self.data,
            yerr=self.noise_map,
            color="k",
            ecolor="k",
            elinewidth=1,
            capsize=2,
        )
        plt.plot(range(self.data.shape[0]), model_data_1d, color="r")
        plt.title("Dynesty model fit to 1D Gaussian + Exponential dataset.")
        plt.xlabel("x values of profile")
        plt.ylabel("Profile normalization")

        os.makedirs(paths.image_path, exist_ok=True)
        plt.savefig(paths.image_path / "model_fit.png")
        plt.clf()

    def visualize_combined(
        self,
        analyses: List[af.Analysis],
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):
        """
        Visualise the instance using images and quantities which are shared across all analyses.

        For example, each Analysis may have a different dataset, where the fit to each dataset is intended to all
        be plotted on the same matplotlib subplot. This function can be overwritten to allow the visualization of such
        a plot.

        Only the first analysis is used to visualize the combined results, where it is assumed that it uses the
        `analyses` property to access the other analyses and perform visualization.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        instance
            The maximum likelihood instance of the model so far in the non-linear search.
        during_analysis
            Is this visualisation during analysis?
        """
        pass

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `pickles` folder such that they can be loaded after the analysis using PyAutoFit's database and
        aggregator tools.

        For this analysis the following are output:

        - The dataset's data.
        - The dataset's noise-map.

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to reperform a fit, this will by default
        load the dataset, settings and other attributes necessary to perform a fit using the attributes output by
        this function.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        """
        paths.save_json(name="data", object_dict=self.data.tolist(), prefix="dataset")
        paths.save_json(name="noise_map", object_dict=self.noise_map.tolist(), prefix="dataset")


