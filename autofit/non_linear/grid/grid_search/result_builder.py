from autofit.non_linear.result import Result
from .result import GridSearchResult


class Placeholder:
    def __getattr__(self, item):
        return None


class ResultBuilder:
    def __init__(self, lists, grid_priors):
        self.lists = lists
        self.grid_priors = grid_priors
        self._job_result_dict = {}

    def __call__(self):
        return GridSearchResult(
            self.results,
            self.lists,
            self.grid_priors
        )

    @property
    def results(self):
        results = []
        for number in range(len(self.lists)):
            try:
                job_result = self._job_result_dict[
                    number
                ]
                results.append(
                    Result(
                        samples=job_result.result.samples,
                        model=job_result.result.model,
                        search=job_result.result.search
                    )
                )
            except KeyError:
                results.append(
                    Placeholder()
                )
        return results

    def add(self, job_result):
        self._job_result_dict[
            job_result.number
        ] = job_result
