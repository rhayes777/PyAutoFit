import csv
import logging
import os
from copy import copy
from itertools import count
from pathlib import Path
from typing import List, Generator, Callable, ClassVar, Union, Tuple

from autoconf.dictable import to_dict
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.grid.grid_search import make_lists
from autofit.non_linear.parallel import AbstractJob, Process, AbstractJobResult
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.result import Result
from autofit.text.text_util import padding


class JobResult(AbstractJobResult):
    def __init__(self, number: int, result: Result, perturb_result: Result):
        """
        The result of a single sensitivity comparison

        Parameters
        ----------
        result
        perturb_result
        """
        super().__init__(number)
        self.result = result
        self.perturb_result = perturb_result

    @property
    def log_likelihood_difference(self):
        return self.perturb_result.log_likelihood - self.result.log_likelihood

    @property
    def log_likelihood_base(self):
        return self.result.log_likelihood

    @property
    def log_likelihood_perturbed(self):
        return self.perturb_result.log_likelihood


class Job(AbstractJob):
    _number = count()

    def __init__(
        self,
        model: AbstractPriorModel,
        simulate_cls: Callable,
        perturb_model: AbstractPriorModel,
        simulate_instance: ModelInstance,
        base_instance: ModelInstance,
        base_fit_cls: Callable,
        perturb_fit_cls: Callable,
        paths: AbstractPaths,
        number: int,
    ):
        """
        Job to run non-linear searches comparing how well a model and a model with a perturbation fit the image.

        Parameters
        ----------
        model
            A base model that fits the image without a perturbation
        perturb_model
            A model of the perturbation which has been added to the underlying image
        base_fit_cls
            A class which defines the function which fits the base model to each simulated dataset of the sensitivity
            map.
        perturb_fit_cls
            A class which defines the function which fits the perturbed model to each simulated dataset of the
            sensitivity map.
        paths
            The paths defining the output directory structure of the sensitivity mapping.
        """
        super().__init__(number=number)

        self.model = model
        self.simulate_cls = simulate_cls
        self.perturb_model = perturb_model
        self.simulate_instance = simulate_instance
        self.base_instance = base_instance
        self.base_fit_cls = base_fit_cls
        self.perturb_fit_cls = perturb_fit_cls
        self.paths = paths

    def perform(self) -> JobResult:
        """
        - Create one model with a perturbation and another without
        - Fit each model against the perturbed image

        Returns
        -------
        An object comprising the results of the two fits
        """

        dataset = self.simulate_cls(
            instance=self.simulate_instance,
            simulate_path=self.paths.image_path.with_name("simulate"),
        )

        result = self.base_fit_cls(
            model=self.model,
            dataset=dataset,
            paths=self.paths.for_sub_analysis("[base]"),
        )

        perturb_model = copy(self.model)
        perturb_model.perturbation = self.perturb_model

        perturb_result = self.perturb_fit_cls(
            model=perturb_model,
            dataset=dataset,
            paths=self.paths.for_sub_analysis("[perturb]"),
        )

        return JobResult(
            number=self.number, result=result, perturb_result=perturb_result
        )


class SensitivityResult:
    def __init__(self, results: List[JobResult]):
        """
        The result of a sensitivity mapping

        Parameters
        ----------
        results
            The results of each sensitivity job
        """
        self.results = sorted(results)

    def __getitem__(self, item):
        return self.results[item]

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    @property
    def log_likelihoods_base(self) -> List[float]:
        """
        The log likelihoods of the base model for each sensitivity fit
        """
        return [result.log_likelihood_base for result in self.results]

    @property
    def log_likelihoods_perturbed(self) -> List[float]:
        """
        The log likelihoods of the perturbed model for each sensitivity fit
        """
        return [result.log_likelihood_perturbed for result in self.results]

    @property
    def log_likelihood_differences(self) -> List[float]:
        """
        The log likelihood differences between the base and perturbed models
        """
        return [result.log_likelihood_difference for result in self.results]


class Sensitivity:
    def __init__(
        self,
        base_model: AbstractPriorModel,
        perturb_model: AbstractPriorModel,
        simulation_instance,
        paths,
        simulate_cls: Callable,
        base_fit_cls: Callable,
        perturb_fit_cls: Callable,
        job_cls: ClassVar = Job,
        number_of_steps: Union[Tuple[int], int] = 4,
        number_of_cores: int = 2,
        limit_scale: int = 1,
    ):
        """
        Perform sensitivity mapping to evaluate whether a perturbation
        can be detected if it occurs in different parts of an image.

        For a range from 0 to 1 with step_size, for each dimension of the
        perturb_model, a perturbation is created and used in conjunction
        with the instance to create an image.

        For each of these images, a fit is run with just the model and with both
        the model and perturb_model to compare how much better the image
        can be fit if the perturbation is included.

        Parameters
        ----------
        base_model
            A model that fits the instance well
        perturb_model
            A model which provides a perturbations to be applied to the instance
            before creating images
        simulation_instance
            An instance of a model to which perturbations are applied prior to
            images being generated
        simulate_cls
            A class which simulates images from each perturb instance that sensitivity mapping is performed on.
        base_fit_cls
            The class which fits the base model to each simulated dataset of the sensitivity map.
        perturb_fit_cls
            The class which fits the perturb model to each simulated dataset of the sensitivity map.
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
        self.logger = logging.getLogger(f"Sensitivity ({paths.name})")

        self.logger.info("Creating")

        self.instance = simulation_instance
        self.model = base_model
        self.perturb_model = perturb_model

        self.paths = paths

        self.simulate_cls = simulate_cls
        self.base_fit_cls = base_fit_cls
        self.perturb_fit_cls = perturb_fit_cls

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
            return tuple(
                [1 / number_of_steps for number_of_steps in self.number_of_steps]
            )
        return 1 / self.number_of_steps

    def run(self) -> SensitivityResult:
        """
        Run fits and comparisons for all perturbations, returning
        a list of results.
        """
        self.logger.info("Running")

        self.paths.save_unique_tag(is_grid_search=True)

        headers = [
            "index",
            *self._headers,
            "log_likelihood_base",
            "log_likelihood_perturbed",
            "log_likelihood_difference",
        ]
        physical_values = list(self._physical_values)

        results = list()
        for result in Process.run_jobs(
            self._make_jobs(), number_of_cores=self.number_of_cores
        ):
            if isinstance(result, Exception):
                raise result

            results.append(result)
            results = sorted(results)

            os.makedirs(self.paths.output_path, exist_ok=True)
            with open(self.results_path, "w+") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for result_ in results:
                    values = physical_values[result_.number]
                    writer.writerow(
                        padding(item)
                        for item in [
                            result_.number,
                            *values,
                            float(result_.log_likelihood_base),
                            float(result_.log_likelihood_perturbed),
                            float(result_.log_likelihood_difference),
                        ]
                    )

        result = SensitivityResult(results)

        self.paths.save_json("result", to_dict(result))

        return SensitivityResult(results)

    @property
    def results_path(self) -> Path:
        return self.paths.output_path / "results.csv"

    @property
    def _lists(self) -> List[List[float]]:
        """
        A list of hypercube vectors, used to instantiate
        the perturb_model and create the individual
        perturbations.
        """
        return make_lists(self.perturb_model.prior_count, step_size=self.step_size)

    @property
    def _physical_values(self) -> List[List[float]]:
        """
        Lists of physical values for each grid square
        """
        return [
            [
                prior.value_for(unit_value)
                for prior, unit_value in zip(
                    self.perturb_model.priors_ordered_by_id, unit_values
                )
            ]
            for unit_values in self._lists
        ]

    @property
    def _headers(self) -> Generator[str, None, None]:
        """
        A name for each of the perturbed priors
        """
        for path, _ in self.perturb_model.prior_tuples:
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
            for value, prior_tuple in zip(list_, self.perturb_model.prior_tuples):
                path, prior = prior_tuple
                value = prior.value_for(value)
                strings.append(f"{path}_{value}")
            yield "_".join(strings)

    @property
    def _perturb_instances(self) -> Generator[ModelInstance, None, None]:
        """
        A list of instances each of which defines a perturbation to
        be applied to the image.
        """
        for list_ in self._lists:
            yield self.perturb_model.instance_from_unit_vector(list_)

    @property
    def _perturb_models(self) -> Generator[AbstractPriorModel, None, None]:
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
                for centre, prior in zip(list_, self.perturb_model.priors_ordered_by_id)
            ]
            yield self.perturb_model.with_limits(limits)

    def _make_jobs(self) -> Generator[Job, None, None]:
        """
        Create a list of jobs to be run on separate processes.

        Each job fits a perturbed image with the original model
        and a model which includes a perturbation.
        """
        for number, (perturb_instance, perturb_model, label) in enumerate(
            zip(self._perturb_instances, self._perturb_models, self._labels)
        ):
            simulate_instance = copy(self.instance)
            simulate_instance.perturb = perturb_instance

            paths = self.paths.for_sub_analysis(
                label,
            )

            yield self.job_cls(
                simulate_instance=simulate_instance,
                model=self.model,
                perturb_model=perturb_model,
                base_instance=self.instance,
                simulate_cls=self.simulate_cls,
                base_fit_cls=self.base_fit_cls,
                perturb_fit_cls=self.perturb_fit_cls,
                paths=paths,
                number=number,
            )
