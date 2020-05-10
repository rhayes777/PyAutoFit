import copy
import logging
import multiprocessing
from time import sleep
from typing import List

import numpy as np

from autofit import conf
from autofit import exc
from autofit.mapper import model_mapper as mm
from autofit.mapper.prior import prior as p
from autofit.optimize import optimizer
from autofit.optimize.non_linear.emcee import Emcee
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.paths import Paths

logger = logging.getLogger(__name__)


class GridSearchResult:
    def __init__(
            self,
            results: List[Result],
            lower_limit_lists: List[List[float]],
            physical_lower_limits_lists: List[List[float]]
    ):
        """
        The result of a grid search.

        Parameters
        ----------
        results
            The results of the non linear optimizations performed at each grid step
        lower_limit_lists
            A list of lists of values representing the lower bounds of the grid searched values at each step
        physical_lower_limits_lists
            A list of lists of values representing the lower physical bounds of the grid search values
            at each step.
        """
        self.lower_limit_lists = lower_limit_lists
        self.physical_lower_limits_lists = physical_lower_limits_lists
        self.results = results
        self.no_dimensions = len(self.lower_limit_lists[0])
        self.no_steps = len(self.lower_limit_lists)
        self.side_length = int(self.no_steps ** (1 / self.no_dimensions))

    def __getattr__(self, item: str) -> object:
        """
        We default to getting attributes from the best result. This allows promises to reference best results.
        """
        return getattr(self.best_result, item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(
            state
        )

    @property
    def shape(self):
        return tuple([self.side_length for dim in range(self.no_dimensions)])

    @property
    def best_result(self):
        """
        The best result of the grid search. That is, the result output by the non linear search that had the highest
        maximum figure of merit.

        Returns
        -------
        best_result: Result
        """
        best_result = None
        for result in self.results:
            if (
                    best_result is None
                    or result.log_likelihood > best_result.log_likelihood
            ):
                best_result = result
        return best_result

    @property
    def best_model(self):
        """
        Returns
        -------
        best_model: mm.ModelMapper
            The model mapper instance associated with the highest figure of merit from the grid search
        """
        return self.best_result.model

    @property
    def all_models(self):
        """
        Returns
        -------
        all_models: [mm.ModelMapper]
            All model mapper instances used in the grid search
        """
        return [result.model for result in self.results]

    @property
    def physical_step_sizes(self):

        physical_step_sizes = []

        for dim in range(self.no_dimensions):
            values = [value[dim] for value in self.physical_lower_limits_lists]
            diff = [abs(values[n] - values[n - 1]) for n in range(1, len(values))]
            physical_step_sizes.append(np.max(diff))

        return tuple(physical_step_sizes)

    @property
    def physical_centres_lists(self):
        return [[lower_limit[dim] + self.physical_step_sizes[dim] / 2 for dim in range(self.no_dimensions)]
                for lower_limit
                in self.physical_lower_limits_lists]

    @property
    def physical_upper_limits_lists(self):
        return [[lower_limit[dim] + self.physical_step_sizes[dim] for dim in range(self.no_dimensions)]
                for lower_limit
                in self.physical_lower_limits_lists]

    @property
    def max_log_likelihood_values(self):
        """
        Returns
        -------
        likelihood_merit_array: np.ndarray
            An arrays of figures of merit. This arrays has the same dimensionality as the grid search, with the value in
            each entry being the figure of merit taken from the optimization performed at that point.
        """
        return np.reshape(
            np.array([result.log_likelihood for result in self.results]),
            tuple(self.side_length for _ in range(self.no_dimensions)),
        )

    @property
    def log_evidence_values(self):
        """
        Returns
        -------
        likelihood_merit_array: np.ndarray
            An arrays of figures of merit. This arrays has the same dimensionality as the grid search, with the value in
            each entry being the figure of merit taken from the optimization performed at that point.
        """
        return np.reshape(
            np.array([result.samples.log_evidence for result in self.results]),
            tuple(self.side_length for _ in range(self.no_dimensions)),
        )


class GridSearch:
    # TODO: this should be using paths
    def __init__(
            self, paths, number_of_steps=4, non_linear_class=Emcee, parallel=False
    ):
        """
        Performs a non linear optimiser search for each square in a grid. The dimensionality of the search depends on
        the number of distinct priors passed to the fit function. (1 / step_size) ^ no_dimension steps are performed
        per an optimisation.

        Parameters
        ----------
        number_of_steps: int
            The number of steps to go in each direction
        non_linear_class: class
            The class of the optimizer that is run at each step
        """
        self.paths = paths

        self.parallel = parallel
        self.number_of_cores = conf.instance.non_linear.config_for(
            "GridSearch"
        ).get(
            "general", "number_of_cores", int
        )
        self.phase_tag_input = paths.phase_tag

        self.number_of_steps = number_of_steps
        self.non_linear_class = non_linear_class

    @property
    def hyper_step_size(self):
        """
        Returns
        -------
        hyper_step_size: float
            The size of a step in any given dimension in hyper space.
        """
        return 1 / self.number_of_steps

    def make_physical_lists(self, grid_priors) -> List[List[float]]:
        lists = self.make_lists(grid_priors)
        return [
            [
                prior.value_for(value)
                for prior, value
                in zip(grid_priors, l)
            ]
            for l in lists
        ]

    def make_lists(self, grid_priors):
        """
        Produces a list of lists of floats, where each list of floats represents the values in each dimension for one
        step of the grid search.

        Parameters
        ----------
        grid_priors: [p.Prior]
            A list of priors that are to be searched using the grid search.

        Returns
        -------
        lists: [[float]]
        """
        return optimizer.make_lists(
            len(grid_priors), step_size=self.hyper_step_size, centre_steps=False
        )

    def make_arguments(self, values, grid_priors):
        arguments = {}
        for value, grid_prior in zip(values, grid_priors):
            if (
                    float("-inf") == grid_prior.lower_limit
                    or float("inf") == grid_prior.upper_limit
            ):
                raise exc.PriorException(
                    "Priors passed to the grid search must have definite limits"
                )
            lower_limit = grid_prior.lower_limit + value * grid_prior.width
            upper_limit = (
                    grid_prior.lower_limit
                    + (value + self.hyper_step_size) * grid_prior.width
            )
            prior = p.UniformPrior(lower_limit=lower_limit, upper_limit=upper_limit)
            arguments[grid_prior] = prior
        return arguments

    def model_mappers(self, model, grid_priors):
        grid_priors = list(set(grid_priors))
        lists = self.make_lists(grid_priors)
        for values in lists:
            arguments = self.make_arguments(values, grid_priors)
            yield model.mapper_from_partial_prior_arguments(arguments)

    def fit(self, model, analysis, grid_priors):
        """
        Fit an analysis with a set of grid priors. The grid priors are priors associated with the model mapper
        of this instance that are replaced by uniform priors for each step of the grid search.

        Parameters
        ----------
        model
        analysis: autofit.optimize.non_linear.non_linear.Analysis
            An analysis used to determine the fitness of a given model instance
        grid_priors: [p.Prior]
            A list of priors to be substituted for uniform priors across the grid.

        Returns
        -------
        result: GridSearchResult
            An object that comprises the results from each individual fit
        """
        if self.parallel:
            return self.fit_parallel(model=model, analysis=analysis, grid_priors=grid_priors)
        else:
            return self.fit_sequential(model=model, analysis=analysis, grid_priors=grid_priors)

    def fit_parallel(self, model, analysis, grid_priors):
        """
        Perform the grid search in parallel, with all the optimisation for each grid square being performed on a
        different process.

        Parameters
        ----------
        analysis
            An analysis
        grid_priors
            Priors describing the position in the grid

        Returns
        -------
        result: GridSearchResult
            The result of the grid search
        """

        grid_priors = list(set(grid_priors))
        results = []
        lists = self.make_lists(grid_priors)
        physical_lists = self.make_physical_lists(
            grid_priors
        )

        results_list = [["index"] +
                        list(map(model.name_for_prior, grid_priors)) + ["likelihood_merit"]
                        ]

        job_queue = multiprocessing.Queue()

        processes = [
            Process(str(number), job_queue)
            for number in range(self.number_of_cores - 1)
        ]

        for index, values in enumerate(lists):
            job = self.job_for_analysis_grid_priors_and_values(
                analysis=copy.deepcopy(analysis),
                model=model,
                grid_priors=grid_priors,
                values=values,
                index=index
            )
            job_queue.put(job)

        for process in processes:
            process.start()

        while len(results) < len(lists):
            for process in processes:
                while not process.queue.empty():
                    result = process.queue.get()
                    results.append(result.result)
                    results_list.append(result.result_list_row)

                    self.write_results(results_list)

        job_queue.close()

        for process in processes:
            process.join(timeout=1.0)

        return GridSearchResult(results, lists, physical_lists)

    def fit_sequential(self, model, analysis, grid_priors):
        """
        Perform the grid search sequentially, with all the optimisation for each grid square being performed on the
        same process.

        Parameters
        ----------
        analysis
            An analysis
        grid_priors
            Priors describing the position in the grid

        Returns
        -------
        result: GridSearchResult
            The result of the grid search
        """

        grid_priors = list(sorted(set(grid_priors), key=lambda prior: prior.id))
        results = []
        lists = self.make_lists(grid_priors)
        physical_lists = self.make_physical_lists(
            grid_priors
        )

        results_list = [["index"] +
                        list(map(model.name_for_prior, grid_priors)) + ["max_log_likelihood"]
                        ]

        for index, values in enumerate(lists):
            job = self.job_for_analysis_grid_priors_and_values(
                analysis=analysis, model=model, grid_priors=grid_priors, values=values, index=index
            )

            result = job.perform()

            results.append(result.result)
            results_list.append(result.result_list_row)

            self.write_results(results_list)

        return GridSearchResult(results, lists, physical_lists)

    def write_results(self, results_list):

        with open("{}/results".format(self.paths.phase_output_path), "w+") as f:
            f.write(
                "\n".join(
                    map(
                        lambda ls: ", ".join(
                            map(
                                lambda value: "{:.2f}".format(value)
                                if isinstance(value, float)
                                else str(value),
                                ls,
                            )
                        ),
                        results_list,
                    )
                )
            )

    def job_for_analysis_grid_priors_and_values(
            self, model, analysis, grid_priors, values, index
    ):
        arguments = self.make_arguments(values=values, grid_priors=grid_priors)
        model = model.mapper_from_partial_prior_arguments(arguments=arguments)

        labels = []
        for prior in sorted(arguments.values(), key=lambda pr: pr.id):
            labels.append(
                "{}_{:.2f}_{:.2f}".format(
                    model.name_for_prior(prior),
                    prior.lower_limit,
                    prior.upper_limit,
                )
            )

        name_path = "{}/{}/{}/{}".format(
            self.paths.phase_name, self.phase_tag_input, self.paths.non_linear_name, "_".join(labels)
        )
        optimizer_instance = self.optimizer_instance(name_path=name_path)

        return Job(
            optimizer_instance=optimizer_instance,
            model=model,
            analysis=analysis,
            arguments=arguments,
            index=index
        )

    def optimizer_instance(self, name_path):

        optimizer_instance = self.non_linear_class(
            Paths(
                phase_name=name_path,
                phase_tag=self.paths.phase_tag,
                phase_folders=self.paths.phase_folders,
                remove_files=self.paths.remove_files,
            )
        )
        for key, value in self.__dict__.items():
            if key not in ("model", "instance", "paths"):
                try:
                    setattr(optimizer_instance, key, value)
                except AttributeError:
                    pass
        return optimizer_instance


class JobResult:
    def __init__(self, result, result_list_row):
        """
        The result of a job

        Parameters
        ----------
        result
            The result of a grid search
        result_list_row
            A row in the result list
        """
        self.result = result
        self.result_list_row = result_list_row


class Job:
    def __init__(self, optimizer_instance, model, analysis, arguments, index):
        """
        A job to be performed in parallel.

        Parameters
        ----------
        optimizer_instance
            An instance of an optimiser
        analysis
            An analysis
        arguments
            The grid search arguments
        """
        self.optimizer_instance = optimizer_instance
        self.analysis = analysis
        self.model = model
        self.arguments = arguments
        self.index = index

    def perform(self):
        result = self.optimizer_instance.full_fit(model=self.model, analysis=self.analysis)
        result_list_row = [self.index, *[prior.lower_limit for prior in self.arguments.values()],
                           result.log_likelihood,
                           ]

        return JobResult(result, result_list_row)


class Process(multiprocessing.Process):
    def __init__(self, name: str, job_queue: multiprocessing.Queue):
        """
        A parallel process that consumes Jobs through the job queue and outputs results through its own queue.

        Parameters
        ----------
        name: str
            The name of the process
        job_queue: multiprocessing.Queue
            The queue through which jobs are submitted
        """
        super().__init__(name=name)
        logger.info("created process {}".format(name))

        self.job_queue = job_queue
        self.queue = multiprocessing.Queue()
        self.count = 0
        self.max_count = 5

    def run(self):
        logger.info("starting process {}".format(self.name))
        while True:
            sleep(0.025)
            if self.count >= self.max_count:
                break
            if self.job_queue.empty():
                self.count += 1
            else:
                self.count = 0
                job = self.job_queue.get()
                self.queue.put(job.perform())
        logger.info("terminating process {}".format(self.name))
        self.job_queue.close()
