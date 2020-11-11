from typing import List, Generator, Callable

from autofit import AbstractPriorModel, ModelInstance, Paths
from autofit.non_linear.parallel import AbstractJob, Process
from .non_linear.grid_search import make_lists


class JobResult:
    pass


class Job(AbstractJob):
    def __init__(
            self,
            instance,
            model,
            perturbation_instance,
            perturbation_model,
            image_function
    ):
        self.instance = instance
        self.model = model
        self.perturbation_instance = perturbation_instance
        self.perturbation_model = perturbation_model
        self.image_function = image_function

    def perform(self):
        image = self.image_function(
            self.instance,
            self.perturbation_instance
        )
        return JobResult()


class Sensitivity:
    def __init__(
            self,
            instance,
            model,
            perturbation_model: AbstractPriorModel,
            image_function: Callable,
            step_size=0.1,
            number_of_cores=2
    ):
        self.instance = instance
        self.model = model
        self.step_size = step_size
        self.perturbation_model = perturbation_model
        self.image_function = image_function
        self.number_of_cores = number_of_cores

    def run(self):
        results = list()
        for result in Process.run_jobs(
                self.make_jobs(),
                number_of_cores=self.number_of_cores
        ):
            results.append(result)
        return results

    @property
    def lists(self) -> List[List[float]]:
        return make_lists(
            self.perturbation_model.prior_count,
            step_size=self.step_size
        )

    @property
    def labels(self):
        for list_ in self.lists:
            strings = list()
            for value, prior_tuple in zip(
                    list_,
                    self.perturbation_model.prior_tuples
            ):
                path, prior = prior_tuple
                value = prior.value_for(
                    value
                )
                strings.append(
                    f"{path}_{value}"
                )
            yield "_".join(strings)

    @property
    def perturbation_instances(self) -> Generator[ModelInstance, None, None]:
        for list_ in self.lists:
            yield self.perturbation_model.instance_from_unit_vector(
                list_
            )

    def make_jobs(self):
        return [
            Job(
                self.instance,
                self.model,
                perturbation_instance,
                self.perturbation_model,
                self.image_function
            )
            for perturbation_instance
            in self.perturbation_instances
        ]
