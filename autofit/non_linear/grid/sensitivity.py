import csv
import logging
import os
from copy import copy
from itertools import count
from pathlib import Path
from typing import List, Generator, Callable, ClassVar, Type, Union, Tuple

from autoconf import cached_property
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.grid.grid_search import make_lists
from autofit.non_linear.parallel import AbstractJob, Process, AbstractJobResult
from autofit.non_linear.result import Result
from autofit.text.text_util import padding


class JobResult(AbstractJobResult):
    def __init__(
            self,
            number: int,
            result: Result,
            perturbed_result: Result
    ):
        """
        The result of a single sensitivity comparison

        Parameters
        ----------
        result
        perturbed_result
        """
        super().__init__(number)
        self.result = result
        self.perturbed_result = perturbed_result

    @property
    def log_likelihood_difference(self):
        return self.perturbed_result.log_likelihood - self.result.log_likelihood

    @property
    def log_likelihood_base(self):
        return self.result.log_likelihood

    @property
    def log_likelihood_perturbed(self):
        return self.perturbed_result.log_likelihood


class Job(AbstractJob):
    _number = count()

    use_instance = False

    def __init__(
            self,
            analysis_factory: "AnalysisFactory",
            model: AbstractPriorModel,
            perturbation_model: AbstractPriorModel,
            base_instance: ModelInstance,
            perturbation_instance: ModelInstance,
            search: NonLinearSearch,
            number: int,
    ):
        """
        Job to run non-linear searches comparing how well a model and a model with a perturbation
        fit the image.

        Parameters
        ----------
        model
            A base model that fits the image without a perturbation
        perturbation_model
            A model of the perturbation which has been added to the underlying image
        analysis_factory
            Factory to generate analysis classes which comprise a model and data
        search
            A non-linear search
        """
        super().__init__(
            number=number
        )

        self.analysis_factory = analysis_factory
        self.model = model

        self.perturbation_model = perturbation_model
        self.base_instance = base_instance
        self.perturbation_instance = perturbation_instance

        self.search = search.copy_with_paths(
            search.paths.for_sub_analysis(
                "[base]",
            )
        )
        self.perturbed_search = search.copy_with_paths(
            search.paths.for_sub_analysis(
                "[perturbed]",
            )
        )

    @cached_property
    def analysis(self):
        return self.analysis_factory()

    def perform(self) -> JobResult:
        """
        - Create one model with a perturbation and another without
        - Fit each model against the perturbed image

        Returns
        -------
        An object comprising the results of the two fits
        """
        result = self.base_model_func()

        perturbed_model = copy(self.model)
        perturbed_model.perturbation = self.perturbation_model

        # TODO : This is what I added so that the Drawer runs use the correct subhalo model.

        # if self.use_instance:
        #     perturbed_model.perturbation = self.perturbation_instance

        perturbed_result = self.perturbation_model_func(perturbed_model=perturbed_model)
        return JobResult(
            number=self.number,
            result=result,
            perturbed_result=perturbed_result
        )

    def base_model_func(self):
        return self.search.fit(
            model=self.model,
            analysis=self.analysis
        )

    def perturbation_model_func(self, perturbed_model):
        return self.perturbed_search.fit(
            model=perturbed_model,
            analysis=self.analysis
        )


class SensitivityResult:

    def __init__(self, results: List[JobResult]):
        self.results = sorted(results)

    def __getitem__(self, item):
        return self.results[item]

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)


class Sensitivity:

    def __init__(
            self,
            base_model: AbstractPriorModel,
            perturbation_model: AbstractPriorModel,
            simulation_instance,
            simulate_function: Callable,
            analysis_class: Type[Analysis],
            search: NonLinearSearch,
            job_cls: ClassVar = Job,
            number_of_steps: Union[Tuple[int], int] = 4,
            number_of_cores: int = 2,
            limit_scale: int = 1,
    ):
        """
        Perform sensitivity mapping to evaluate whether a perturbation
        can be detected if it occurs in different parts of an image.

        For a range from 0 to 1 with step_size, for each dimension of the
        perturbation_model, a perturbation is created and used in conjunction
        with the instance to create an image.

        For each of these images, a fit is run with just the model and with both
        the model and perturbation_model to compare how much better the image
        can be fit if the perturbation is included.

        Parameters
        ----------
        base_model
            A model that fits the instance well
        perturbation_model
            A model which provides a perturbations to be applied to the instance
            before creating images
        simulation_instance
            An instance of a model to which perturbations are applied prior to
            images being generated
        simulate_function
            A function that can convert an instance into an image
        analysis_class
            A class which can compare an image to an instance and evaluate fitness
        search
            A NonLinear search class which is copied and used to evaluate fitness
        number_of_cores
            How many cores does this computer have? Minimum 2.
        limit_scale
            Scales the priors for each perturbation model.
                A scale of 1 means priors have limits the same size as the grid square.
                A scale of 2 means priors have limits larger than the grid square with
                    width twice a grid square.
                A scale of 0.5 means priors have limits smaller than the grid square
                    with width half a grid square.
        """
        self.logger = logging.getLogger(
            f"Sensitivity ({search.name})"
        )

        self.logger.info("Creating")

        self.instance = simulation_instance
        self.model = base_model

        self.search = search
        self.analysis_class = analysis_class

        self.perturbation_model = perturbation_model
        self.simulate_function = simulate_function

        self.job_cls = job_cls

        self.number_of_steps = number_of_steps
        self.number_of_cores = number_of_cores or 2

        self.limit_scale = limit_scale

    @property
    def step_size(self):
        """
        Returns
        -------
        step_size: float
            The size of a step in any given dimension in hyper space.
        """
        if isinstance(self.number_of_steps, tuple):
            return tuple([1 / number_of_steps for number_of_steps in self.number_of_steps])
        return 1 / self.number_of_steps

    def run(self) -> SensitivityResult:
        """
        Run fits and comparisons for all perturbations, returning
        a list of results.
        """
        self.logger.info("Running")

        headers = [
            "index",
            *self._headers,
            "log_likelihood_base",
            "log_likelihood_perturbed",
            "log_likelihood_difference"
        ]
        physical_values = list(self._physical_values)

        results = list()
        for result in Process.run_jobs(
                self._make_jobs(),
                number_of_cores=self.number_of_cores
        ):
            if isinstance(result, Exception):
                raise result

            results.append(result)
            results = sorted(results)

            os.makedirs(
                self.search.paths.output_path,
                exist_ok=True
            )
            with open(self.results_path, "w+") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for result_ in results:
                    values = physical_values[
                        result_.number
                    ]
                    writer.writerow(
                        padding(item)
                        for item in [
                            result_.number,
                            *values,
                            result_.log_likelihood_base,
                            result_.log_likelihood_perturbed,
                            result_.log_likelihood_difference,
                        ])

        return SensitivityResult(results)

    @property
    def results_path(self):
        return Path(
            self.search.paths.output_path
        ) / "results.csv"

    @property
    def _lists(self) -> List[List[float]]:
        """
        A list of hypercube vectors, used to instantiate
        the perturbation_model and create the individual
        perturbations.
        """
        return make_lists(
            self.perturbation_model.prior_count,
            step_size=self.step_size
        )

    @property
    def _physical_values(self) -> List[List[float]]:
        """
        Lists of physical values for each grid square
        """
        return [
            [
                prior.value_for(
                    unit_value
                )
                for prior, unit_value
                in zip(
                self.perturbation_model.priors_ordered_by_id,
                unit_values
            )
            ]
            for unit_values in self._lists
        ]

    @property
    def _headers(self) -> Generator[str, None, None]:
        """
        A name for each of the perturbed priors
        """
        for path, _ in self.perturbation_model.prior_tuples:
            yield path

    @property
    def _labels(self) -> Generator[str, None, None]:
        """
        One label for each perturbation, used to distinguish
        fits for each perturbation by placing them in separate
        directories.
        """
        for list_ in self._lists:
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
    def _perturbation_instances(self) -> Generator[
        ModelInstance, None, None
    ]:
        """
        A list of instances each of which defines a perturbation to
        be applied to the image.
        """
        for list_ in self._lists:
            yield self.perturbation_model.instance_from_unit_vector(
                list_
            )

    @property
    def _perturbation_models(self) -> Generator[
        AbstractPriorModel, None, None
    ]:
        """
        A list of models representing a perturbation at each grid square.

        By default models have priors with limits at the edges of a grid square.
        These limits can be scaled using the limit_scale variable. If the variable
        is 2 then the priors will have width twice the step size.
        """
        half_step = self.limit_scale * self.step_size / 2
        for list_ in self._lists:
            limits = [
                (
                    prior.value_for(max(0.0, centre - half_step)),
                    prior.value_for(min(1.0, centre + half_step)),
                )
                for centre, prior in zip(
                    list_,
                    self.perturbation_model.priors_ordered_by_id
                )
            ]
            yield self.perturbation_model.with_limits(limits)

    @property
    def _searches(self) -> Generator[
        NonLinearSearch, None, None
    ]:
        """
        A list of non-linear searches, each of which is applied to
        one perturbation.
        """
        for label in self._labels:
            yield self._search_instance(
                label
            )

    def _search_instance(
            self,
            name_path: str
    ) -> NonLinearSearch:
        """
        Create a search instance, distinguished by its name

        Parameters
        ----------
        name_path
            A path to distinguish this search from other searches

        Returns
        -------
        A non linear search, copied from the instance search
        """
        paths = self.search.paths
        search_instance = self.search.copy_with_paths(
            paths.for_sub_analysis(
                name_path,
            )
        )

        return search_instance

    def _make_jobs(self) -> Generator[Job, None, None]:
        """
        Create a list of jobs to be run on separate processes.

        Each job fits a perturbed image with the original model
        and a model which includes a perturbation.
        """
        for number, (
                perturbation_instance,
                perturbation_model,
                search
        ) in enumerate(zip(
            self._perturbation_instances,
            self._perturbation_models,
            self._searches
        )):
            instance = copy(self.instance)
            instance.perturbation = perturbation_instance

            yield self.job_cls(
                analysis_factory=AnalysisFactory(
                    instance=instance,
                    simulate_function=self.simulate_function,
                    analysis_class=self.analysis_class,
                ),
                model=self.model,
                perturbation_model=perturbation_model,
                base_instance=self.instance,
                perturbation_instance=perturbation_instance,
                search=search,
                number=number
            )


class AnalysisFactory:
    def __init__(
            self,
            instance,
            simulate_function,
            analysis_class,
    ):
        """
        Callable to delay simulation such that it is performed
        on the Job subprocess
        """
        self.instance = instance
        self.simulate_function = simulate_function
        self.analysis_class = analysis_class

    def __call__(self):
        dataset = self.simulate_function(
            self.instance
        )
        return self.analysis_class(
            dataset
        )
