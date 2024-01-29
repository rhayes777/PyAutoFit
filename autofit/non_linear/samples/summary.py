from typing import Optional

import numpy as np

from .interface import SamplesInterface, apply_derived_quantities
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from .sample import Sample


class SamplesSummary(SamplesInterface):
    def __init__(
        self,
        max_log_likelihood_sample: Sample,
        model: AbstractPriorModel,
        covariance_matrix: Optional[np.ndarray] = None,
        log_evidence: Optional[float] = None,
    ):
        """
        A summary of the results of a `NonLinearSearch` that has been run, including the maximum log likelihood

        Parameters
        ----------
        max_log_likelihood_sample
            The parameters from a non-linear search that gave the highest likelihood
        model
            A model used to map the samples to physical values
        covariance_matrix
            The covariance matrix of the samples
        """
        super().__init__(model=model)
        self._max_log_likelihood_sample = max_log_likelihood_sample
        self.covariance_matrix = covariance_matrix
        self._log_evidence = log_evidence
        self.derived_summary = None

    @property
    def max_log_likelihood_sample(self):
        return self._max_log_likelihood_sample

    @property
    def log_evidence(self):
        return self._log_evidence

    def max_log_likelihood(self, as_instance=True):
        instance = super().max_log_likelihood(as_instance=as_instance)
        try:
            derived_quantities = {
                tuple(key.split(".")): value
                for key, value in self.derived_summary[
                    "max_log_likelihood_sample"
                ].items()
            }
            apply_derived_quantities(instance, derived_quantities)

        except (KeyError, TypeError):
            pass
        return instance
