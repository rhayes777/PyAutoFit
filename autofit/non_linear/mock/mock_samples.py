from typing import Optional

from autofit.non_linear.samples import PDFSamples, Sample, NestSamples



def samples_with_log_likelihood_list(
        log_likelihood_list
):
    return [
        Sample(
            log_likelihood=log_likelihood,
            log_prior=0,
            weight=0
        )
        for log_likelihood
        in log_likelihood_list
    ]


class MockSamples(PDFSamples):
    def __init__(
            self,
            model=None,
            sample_list=None,
            max_log_likelihood_instance=None,
            log_likelihood_list=None,
            gaussian_tuples=None,
            unconverged_sample_size=10,
            **kwargs,
    ):

        self._log_likelihood_list = log_likelihood_list

        self.model = model

        sample_list = sample_list or self.default_sample_list

        super().__init__(
            model=model, sample_list=sample_list, unconverged_sample_size=unconverged_sample_size, **kwargs
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
            Sample(
                log_likelihood=log_likelihood,
                log_prior=0.0,
                weight=0.0
            )
            for log_likelihood
            in log_likelihood_list
        ]

    @property
    def log_likelihood_list(self):

        if self._log_likelihood_list is None:
            return super().log_likelihood_list

        return self._log_likelihood_list

    @property
    def max_log_likelihood_instance(self):

        if self._max_log_likelihood_instance is None:

            try:
                return super().max_log_likelihood_instance
            except (KeyError, AttributeError):
                pass

        return self._max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma=None):

        if self._gaussian_tuples is None:
            return super().gaussian_priors_at_sigma(sigma=sigma)

        return self._gaussian_tuples

    def write_table(self, filename):
        pass


class MockNestSamples(NestSamples):

    def __init__(
            self,
            model,
            sample_list=None,
            total_samples=10,
            log_evidence=0.0,
            number_live_points=5,
            time: Optional[float] = None,
    ):

        self.model = model

        if sample_list is None:

            sample_list = [
                Sample(
                    log_likelihood=log_likelihood,
                    log_prior=0.0,
                    weight=0.0
                )
                for log_likelihood
                in self.log_likelihood_list
            ]

        super().__init__(
            model=model,
            sample_list=sample_list,
            time=time
        )

        self._total_samples = total_samples
        self._log_evidence = log_evidence
        self._number_live_points = number_live_points

    @property
    def total_samples(self):
        return self._total_samples

    @property
    def log_evidence(self):
        return self._log_evidence

    @property
    def number_live_points(self):
        return self._number_live_points



