from autofit.mapper.model import ModelInstance
from autofit.mapper.model_mapper import ModelMapper
from autofit.non_linear.result import Result

from autofit.non_linear.mock.mock_samples import MockSamples


class MockResult(Result):
    def __init__(
            self,
            samples=None,
            instance=None,
            model=None,
            analysis=None,
            search=None,
    ):

        super().__init__(samples, model, search)

        self._instance = instance or ModelInstance()
        self.model = model or ModelMapper()
        self.samples = samples or MockSamples(max_log_likelihood_instance=self.instance)

        self.gaussian_tuples = None
        self.analysis = analysis
        self.search = search
        self.model = model

    def model_absolute(self, absolute):
        return self.model

    def model_relative(self, relative):
        return self.model

    @property
    def last(self):
        return self



class MockResultGrid(Result):
    def __init__(self, log_likelihood):
        # noinspection PyTypeChecker
        super().__init__(None, None)
        self._log_likelihood = log_likelihood
        self.model = log_likelihood

    @property
    def log_likelihood(self):
        return self._log_likelihood
