from typing import Optional, Union

from autofit.non_linear.samples import SamplesPDF, Sample, SamplesNest


def samples_with_log_likelihood_list(log_likelihood_list):
    return [
        Sample(log_likelihood=log_likelihood, log_prior=0, weight=0)
        for log_likelihood in log_likelihood_list
    ]


class MockSamples(SamplesPDF):
    def __init__(
        self,
        model=None,
        sample_list=None,
        samples_info=None,
        max_log_likelihood_instance=None,
        log_likelihood_list=None,
        gaussian_tuples=None,
        **kwargs,
    ):
        self._log_likelihood_list = log_likelihood_list

        self.model = model

        sample_list = sample_list or self.default_sample_list

        samples_info = samples_info or {"unconverged_sample_size": 0}

        super().__init__(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
            **kwargs,
        )

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self._gaussian_tuples = gaussian_tuples

    @property
    def default_sample_list(self):
        if self._log_likelihood_list is not None:
            log_likelihood_list = self._log_likelihood_list
        else:
            log_likelihood_list = range(3)

        return [
            Sample(log_likelihood=log_likelihood, log_prior=0.0, weight=0.0)
            for log_likelihood in log_likelihood_list
        ]

    @property
    def log_likelihood_list(self):
        if self._log_likelihood_list is None:
            return super().log_likelihood_list

        return self._log_likelihood_list

    def max_log_likelihood(self, as_instance: bool = True):
        if self._max_log_likelihood_instance is None:
            try:
                return super().max_log_likelihood(as_instance=as_instance)
            except (KeyError, AttributeError):
                pass

        return self._max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma=None):

        if self._gaussian_tuples is None:
            return super().gaussian_priors_at_sigma(sigma=sigma)

        return self._gaussian_tuples

    @property
    def unconverged_sample_size(self):
        return self.samples_info["unconverged_sample_size"]


class MockSamplesNest(SamplesNest):
    def __init__(
        self,
        model,
        sample_list=None,
        samples_info=None,
    ):
        self.model = model

        if sample_list is None:
            sample_list = [
                Sample(log_likelihood=log_likelihood, log_prior=0.0, weight=0.0)
                for log_likelihood in self.log_likelihood_list
            ]

        super().__init__(model=model, sample_list=sample_list, samples_info=samples_info)

