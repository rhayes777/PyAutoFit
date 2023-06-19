from .interface import SamplesInterface
from ...tools.util import to_dict


class SamplesSummary(SamplesInterface):
    def __init__(self, max_log_likelihood_sample, model, covariance_matrix=None):
        self.max_log_likelihood_sample = max_log_likelihood_sample
        self.model = model
        self.covariance_matrix = covariance_matrix

    def dict(self):
        return {
            "max_log_likelihood_sample": to_dict(self.max_log_likelihood_sample),
            "model": self.model.dict(),
            "covariance_matrix": self.covariance_matrix.tolist(),
        }
