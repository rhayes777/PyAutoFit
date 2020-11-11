from typing import List, Generator, Callable

from autofit import AbstractPriorModel, ModelInstance, Paths, CollectionPriorModel, Result
from autofit.non_linear.parallel import AbstractJob, Process
from .non_linear.grid_search import make_lists


class JobResult:
    def __init__(
            self,
            result: Result,
            perturbed_result: Result
    ):
        self.result = result
        self.perturbed_result = perturbed_result


class Job(AbstractJob):
    def __init__(
            self,
            image,
            model,
            perturbation_model,
            analysis_class,
            search
    ):
        self.image = image
        self.model = model

        self.perturbation_model = perturbation_model
        self.analysis_class = analysis_class

        paths = search.paths

        self.search = search
        self.perturbed_search = search.copy_with_paths(
            Paths(
                name=paths.name,
                tag=paths.tag + "[perturbed]",
                path_prefix=paths.path_prefix,
                remove_files=paths.remove_files,
            )
        )

    @property
    def analysis(self):
        return self.analysis_class(
            self.image
        )

    def perform(self):
        model = CollectionPriorModel()
        model.model = self.model
        result = self.search.fit(
            model=self.model,
            analysis=self.analysis
        )

        perturbed_model = CollectionPriorModel()
        perturbed_model.perturbation = self.perturbation_model
        perturbed_model.model = self.model

        perturbed_result = self.perturbed_search.fit(
            model=perturbed_model,
            analysis=self.analysis
        )
        return JobResult(
            result=result,
            perturbed_result=perturbed_result
        )


class Sensitivity:
    def __init__(
            self,
            instance,
            model,
            search,
            analysis_class,
            perturbation_model: AbstractPriorModel,
            image_function: Callable,
            step_size=0.1,
            number_of_cores=2
    ):
        self.instance = instance
        self.model = model

        self.search = search
        self.analysis_class = analysis_class

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

    @property
    def searches(self):
        for label in self.labels:
            paths = self.search.paths
            name_path = "{}/{}/{}/{}".format(
                paths.name,
                paths.tag,
                paths.non_linear_tag,
                label,
            )
            yield self.search_instance(
                name_path
            )

    def search_instance(self, name_path):
        paths = self.search.paths
        search_instance = self.search.copy_with_paths(
            Paths(
                name=name_path,
                tag=paths.tag,
                path_prefix=paths.path_prefix,
                remove_files=paths.remove_files,
            )
        )

        return search_instance

    def make_jobs(self):
        for perturbation_instance, search in zip(
                self.perturbation_instances,
                self.searches
        ):
            instance = ModelInstance()
            instance.model = self.instance
            instance.perturbation = perturbation_instance
            image = self.image_function(
                instance
            )
            yield Job(
                image=image,
                model=self.model,
                perturbation_model=self.perturbation_model,
                search=search,
                analysis_class=self.analysis_class
            )
