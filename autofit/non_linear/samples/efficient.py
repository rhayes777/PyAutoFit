import numpy as np
from .sample import Sample
from .pdf import SamplesPDF


class EfficientSamples:
    def __init__(self, samples: SamplesPDF):
        self.model = samples.model
        self.samples_info = samples.samples_info
        self.search_internal = samples.search_internal

        sample_list = samples.sample_list
        self._kwargs = sample_list[0].kwargs
        self._log_likelihoods = np.asarray(
            [sample.log_likelihood for sample in sample_list]
        )
        self._log_priors = np.asarray([sample.log_prior for sample in sample_list])
        self._weights = np.asarray([sample.weight for sample in sample_list])

    @property
    def samples(self):
        return SamplesPDF(
            model=self.model,
            samples_info=self.samples_info,
            search_internal=self.search_internal,
            sample_list=self.sample_list,
        )

    @property
    def sample_list(self):
        return [
            Sample(
                log_likelihood=log_likelihood,
                log_prior=log_prior,
                weight=weight,
                kwargs=self._kwargs,
            )
            for log_likelihood, log_prior, weight in zip(
                self._log_likelihoods, self._log_priors, self._weights
            )
        ]
