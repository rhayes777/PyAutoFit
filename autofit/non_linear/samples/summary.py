from typing import Optional
import logging

import numpy as np

from .interface import SamplesInterface
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from .sample import Sample


logger = logging.getLogger(__name__)


class SamplesSummary(SamplesInterface):
    def __init__(
        self,
        model: AbstractPriorModel,
        max_log_likelihood_sample: Sample,
        median_pdf : Optional[Sample] = None,
        log_evidence: Optional[float] = None,
    ):
        """
        A summary of the results of a `NonLinearSearch` that has been run, including the maximum log likelihood

        Parameters
        ----------
        model
            A model used to map the samples to physical values
        max_log_likelihood_sample
            The parameters from a non-linear search that gave the highest likelihood
        median_pdf
            The median PDF of the samples which are used for prior linking via the search chaining API.
        """
        super().__init__(model=model)

        self._max_log_likelihood_sample = max_log_likelihood_sample
        self.median_pdf = median_pdf
        self._log_evidence = log_evidence
        self.derived_summary = None

    @property
    def max_log_likelihood_sample(self):
        return self._max_log_likelihood_sample

    @property
    def log_evidence(self):
        return self._log_evidence
