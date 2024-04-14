from typing import List, Optional
import logging

import numpy as np

from .interface import SamplesInterface
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from .sample import Sample

from autofit.non_linear.samples.interface import to_instance


logger = logging.getLogger(__name__)


class SamplesSummary(SamplesInterface):
    def __init__(
        self,
        max_log_likelihood_sample: Sample,
        model: AbstractPriorModel = None,
        median_pdf_sample : Optional[Sample] = None,
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
        median_pdf_sample
            The median PDF of the samples which are used for prior linking via the search chaining API.
        """
        super().__init__(model=model)

        self._max_log_likelihood_sample = max_log_likelihood_sample
        self._median_pdf_sample = median_pdf_sample
        self._log_evidence = log_evidence
        self.derived_summary = None

    @property
    def max_log_likelihood_sample(self):
        return self._max_log_likelihood_sample

    @property
    def median_pdf_sample(self):
        return self._median_pdf_sample

    @to_instance
    def median_pdf(self, as_instance: bool = True) -> List[float]:
        """
        The parameters of the maximum log likelihood sample of the `NonLinearSearch` returned as a model instance or
        list of values.
        """

        sample = self.median_pdf_sample

        print(sample)
        vvv

        return sample.parameter_lists_for_paths(
            self.paths if sample.is_path_kwargs else self.names
        )

    @property
    def log_evidence(self):
        return self._log_evidence
