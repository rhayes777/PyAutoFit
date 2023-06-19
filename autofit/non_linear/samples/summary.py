import numpy as np

from .interface import SamplesInterface
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from ...tools.util import to_dict, from_dict


class SamplesSummary(SamplesInterface):
    def __init__(self, max_log_likelihood_sample, model, covariance_matrix=None):
        self.max_log_likelihood_sample = max_log_likelihood_sample
        self.model = model
        self.covariance_matrix = covariance_matrix

    def dict(self):
        return {
            "max_log_likelihood_sample": to_dict(self.max_log_likelihood_sample),
            "model": self.model.dict(),
            "covariance_matrix": self.covariance_matrix.tolist()
            if self.covariance_matrix is not None
            else None,
        }

    @classmethod
    def from_dict(cls, summary_dict):
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
