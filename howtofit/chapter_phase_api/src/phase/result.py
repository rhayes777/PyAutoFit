import autofit as af
from src.phase.analysis import Analysis

import numpy as np

"""
The Result class stores the results of a `NonLinearSearch`, which are inherited from PyAutoFit`s Result object. In
chapter 1 we described the results this object contains.

By using the PyAutoFit phase API, we are able to extend the Result object returned by a phase`s `NonLinearSearch` to
include model-specific properties of the fit. Below, we include method that return the model data and fit of the
maximum log likelihood model, which we use in the main tutorial script to plot the results. In chapter 1 we had to
create these manually; the phase API means we can provide these convenience methods for our users.
"""


class Result(af.Result):
    def __init__(
        self,
        samples: af.PDFSamples,
        previous_model: af.ModelMapper,
        search: af.NonLinearSearch,
        analysis: Analysis,
    ):
        """
        The results of a `NonLinearSearch` performed by a phase.

        Parameters
        ----------
        samples : af.Samples
            A class containing the samples of the `NonLinearSearch`, including methods to get the maximum log
            likelihood model, errors, etc.
        analysis : Analysis
            The Analysis class used by this model-fit to fit the model to the data.
        """
        super().__init__(samples=samples, previous_model=previous_model, search=search)
        self.analysis = analysis

    @property
    def max_log_likelihood_model_data(self) -> np.ndarray:
        return self.analysis.model_data_from_instance(
            instance=self.samples.max_log_likelihood_instance
        )

    @property
    def max_log_likelihood_fit(self) -> np.ndarray:
        return self.analysis.fit_from_model_data(
            model_data=self.max_log_likelihood_model_data
        )
