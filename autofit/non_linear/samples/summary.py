import numpy as np

from .interface import SamplesInterface
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.tools.util import to_dict, from_dict
from .sample import Sample


class SamplesSummary(Sample1sInterface):
    def __init__(
        self,
        max_log_likelihood_sample: Sample,
        model: AbstractPriorModel,
        covariance_matrix: np.ndarray = None,
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
        self._covariance_matrix = covariance_matrix

    def covariance_matrix(self) -> np.ndarray:
        return self._covariance_matrix

    def dict(self) -> dict:
        """
        A JSON serialisable dictionary representation of the summary
        """
        return {
            "max_log_likelihood_sample": to_dict(self.max_log_likelihood_sample),
            "model": self.model.dict(),
            "covariance_matrix": self.covariance_matrix().tolist()
            if self.covariance_matrix is not None
            else None,
        }

    @classmethod
    def from_dict(cls, summary_dict: dict) -> "SamplesSummary":
        """
        Create a summary from a dictionary representation
        """
        try:
            covariance_matrix = np.array(summary_dict["covariance_matrix"])
        except (KeyError, ValueError):
            covariance_matrix = None
        return cls(
            max_log_likelihood_sample=from_dict(
                summary_dict["max_log_likelihood_sample"]
            ),
            model=AbstractPriorModel.from_dict(summary_dict["model"]),
            covariance_matrix=covariance_matrix,
        )

    @property
    def max_log_likelihood_sample(self):
        return self._max_log_likelihood_sample
