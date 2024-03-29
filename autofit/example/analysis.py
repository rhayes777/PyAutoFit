import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from autofit.example.result import ResultExample
from autofit.jax_wrapper import numpy as np

import autofit as af

"""
The `analysis.py` module contains the dataset and log likelihood function which given a model instance (set up by
the non-linear search) fits the dataset and returns the log likelihood of that model.
"""

class Analysis(af.Analysis):

    """
    This is the result object returned by a fit using this analysis class.

    This result has been extended, based on the model that is input into the analysis, to include a property
    `max_log_likelihood_model_data`, which is the model data of the best-fit model.
    """
    Result = ResultExample

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
        model_data_1d = self.model_data_1d_from(instance=instance)

        residual_map = self.data - model_data_1d
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)

        return log_likelihood

    def model_data_1d_from(self, instance : af.ModelInstance) -> np.ndarray:
        """
        Returns the model data of a the 1D profiles.

        The way this is generated changes depending on if the model is a `Model` (therefore having only one profile)
        or a `Collection` (therefore having multiple profiles).

        If its a model, the model component's `model_data_1d_via_xvalues_from` is called and the output returned.
        For a collection, each components `model_data_1d_via_xvalues_from` is called, iterated through and summed
        to return the combined model data.

        Parameters
        ----------
        instance
            The model instance of the profile or collection of profiles.

        Returns
        -------
        The model data of the profiles.
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

        return model_data_1d

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
        plt.close()

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


    def make_result(
        self,
        samples_summary: af.SamplesSummary,
        paths: af.AbstractPaths,
        samples: Optional[af.SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[object] = None,
    ) -> Result:
        """
        Returns the `Result` of the non-linear search after it is completed.

        The result type is defined as a class variable in the `Analysis` class (see top of code under the python code
        `class Analysis(af.Analysis)`.

        The result can be manually overwritten by a user to return a user-defined result object, which can be extended
        with additional methods and attribute specific to the model-fit.

        This example class does example this, whereby the analysis result has been over written with the `ResultExample`
        class, which contains a property `max_log_likelihood_model_data_1d` that returns the model data of the
        best-fit model. This API means you can customize your result object to include whatever attributes you want
        and therefore make a result object specific to your model-fit and model-fitting problem.

        The `Result` object you return can be customized to include:

        - The samples summary, which contains the maximum log likelihood instance and median PDF model.

        - The paths of the search, which are used for loading the samples and search internal below when a search
        is resumed.

        - The samples of the non-linear search (e.g. MCMC chains) also stored in `samples.csv`.

        - The non-linear search used for the fit in its internal representation, which is used for resuming a search
        and making bespoke visualization using the search's internal results.

        - The analysis used to fit the model (default disabled to save memory, but option may be useful for certain
        projects).

        Parameters
        ----------
        samples_summary
            The summary of the samples of the non-linear search, which include the maximum log likelihood instance and
            median PDF model.
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        samples
            The samples of the non-linear search, for example the chains of an MCMC run.
        search_internal
            The internal representation of the non-linear search used to perform the model-fit.
        analysis
            The analysis used to fit the model.

        Returns
        -------
        Result
            The result of the non-linear search, which is defined as a class variable in the `Analysis` class.
        """
        return self.Result(
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=None
        )

    def compute_latent_variable(self, instance) -> Dict[str, float]:
        """
        A latent variable is not a model parameter but can be derived from the model. Its value and errors may be
        of interest and aid in the interpretation of a model-fit.

        For example, for the simple 1D Gaussian example, it could be the full-width half maximum (FWHM) of the
        Gaussian. This is not included in the model but can be easily derived from the Gaussian's sigma value.

        By overwriting this method we can manually specify latent variables that are calculated and output to
        a `latent.csv` file, which mirrors the `samples.csv` file.

        In the example below, the `latent.csv` file will contain one column with the FWHM of every Gausian model
        sampled by the non-linear search.

        This function is called for every non-linear search sample, where the `instance` passed in corresponds to
        each sample.

        Parameters
        ----------
        instance
            The instances of the model which the latent variable is derived from.

        Returns
        -------

        """
        try:
            return {
                "fwhm": instance.fwhm
            }
        except AttributeError:
            return {}