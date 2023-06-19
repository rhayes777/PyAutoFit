from .interface import SamplesInterface


class SamplesSummary(SamplesInterface):
    def __init__(self, max_log_likelihood_sample, model, covariance_matrix=None):
        self.max_log_likelihood_sample = max_log_likelihood_sample
        self.model = model
        self.covariance_matrix = covariance_matrix
