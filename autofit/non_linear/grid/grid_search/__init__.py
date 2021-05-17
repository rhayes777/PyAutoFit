import copy
from os import path
from typing import List, Tuple, Union

from autofit import exc
from autofit.mapper.prior import prior as p
from autofit.non_linear.parallel import Process
from .job import Job
from .result import GridSearchResult


class GridSearch:

    def __init__(self, search, number_of_steps=4, number_of_cores=1):
        """
        Performs a non linear optimiser search for each square in a grid. The dimensionality of the search depends on
        the number of distinct priors passed to the fit function. (1 / step_size) ^ no_dimension steps are performed
        per an optimisation.

        Parameters
        ----------
        number_of_steps: int
            The number of steps to go in each direction
        search: class
            The class of the search that is run at each step
        """
        self.paths = search.paths

        self.number_of_cores = number_of_cores or 1

        if self.number_of_cores == 1:
            self.parallel = False
        else:
            self.parallel = True

        self.number_of_steps = number_of_steps
        self.search = search
        self.prior_passer = search.prior_passer

    @property
    def step_size(self):
        """
        Returns
        -------
        step_size: float
            The size of a step in any given dimension in hyper space.
        """
        return 1 / self.number_of_steps

    def make_physical_lists(self, grid_priors) -> List[List[float]]:
        lists = self.make_lists(grid_priors)
        return [
            [prior.value_for(value) for prior, value in zip(grid_priors, l)]
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
        return make_lists(
            len(grid_priors), step_size=self.step_size, centre_steps=False
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
                    + (value + self.step_size) * grid_prior.width
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
        analysis: autofit.non_linear.non_linear.Analysis
            An analysis used to determine the fitness of a given model instance
        grid_priors: [p.Prior]
            A list of priors to be substituted for uniform priors across the grid.

        Returns
        -------
        result: GridSearchResult
            An object that comprises the results from each individual fit
        """
        func = self.fit_parallel if self.parallel else self.fit_sequential
        return func(
            model=model,
            analysis=analysis,
            grid_priors=grid_priors
        )

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
        physical_lists = self.make_physical_lists(grid_priors)

        results_list = [
            ["index"]
            + list(map(model.name_for_prior, grid_priors))
            + ["likelihood_merit"]
        ]

        for result in Process.run_jobs(
                self.make_jobs(
                    model,
                    analysis,
                    grid_priors
                ),
                self.number_of_cores
        ):
            results.append(result)
            results = sorted(results)
            results_list.append(result.result_list_row)
            self.write_results(results_list)

        return GridSearchResult(
            [
                result.result
                for result
                in results
            ],
            lists,
            physical_lists
        )

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
        physical_lists = self.make_physical_lists(grid_priors)

        results_list = [
            ["index"]
            + list(map(model.name_for_prior, grid_priors))
            + ["max_log_likelihood"]
        ]

        for job in self.make_jobs(
                model,
                analysis,
                grid_priors
        ):

            result = job.perform()

            results.append(result.result)
            results_list.append(result.result_list_row)

            self.write_results(results_list)

        return GridSearchResult(results, lists, physical_lists)

    def make_jobs(self, model, analysis, grid_priors):
        grid_priors = list(set(grid_priors))
        lists = self.make_lists(grid_priors)

        jobs = list()

        for index, values in enumerate(lists):
            jobs.append(
                self.job_for_analysis_grid_priors_and_values(
                    analysis=copy.deepcopy(analysis),
                    model=model,
                    grid_priors=grid_priors,
                    values=values,
                    index=index,
                )
            )
        return jobs

    def write_results(self, results_list):

        with open(path.join(self.paths.output_path, "results"), "w+") as f:
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
        self.paths.model = model
        self.paths.search = self

        arguments = self.make_arguments(values=values, grid_priors=grid_priors)
        model = model.mapper_from_partial_prior_arguments(arguments=arguments)

        labels = []
        for prior in sorted(arguments.values(), key=lambda pr: pr.id):
            labels.append(
                "{}_{:.2f}_{:.2f}".format(
                    model.name_for_prior(prior), prior.lower_limit, prior.upper_limit
                )
            )

        name_path = path.join(
            self.paths.name,
            self.paths.identifier,
            "_".join(labels),
        )

        search_instance = self.search_instance(name_path=name_path)
        search_instance.paths.model = model

        return Job(
            search_instance=search_instance,
            model=model,
            analysis=analysis,
            arguments=arguments,
            index=index,
        )

    def search_instance(self, name_path):
        search_instance = self.search.copy_with_paths(
            self.paths.create_child(
                name=name_path,
                path_prefix=self.paths.path_prefix,
                is_identifier_in_paths=False
            )
        )

        for key, value in self.__dict__.items():
            if key not in ("model", "instance", "paths"):
                try:
                    setattr(search_instance, key, value)
                except AttributeError:
                    pass
        return search_instance


def grid(fitness_function, no_dimensions, step_size):
    """
    Grid2D search using a fitness function over a given number of dimensions and a given step size between inclusive
    limits of 0 and 1.

    Parameters
    ----------
    fitness_function: function
        A function that takes a tuple of floats as an argument
    no_dimensions: int
        The number of dimensions of the grid search
    step_size: float
        The step size of the grid search

    Returns
    -------
    best_arguments: tuple[float]
        The tuple of arguments that gave the highest fitness
    """
    best_fitness = float("-inf")
    best_arguments = None

    for arguments in make_lists(no_dimensions, step_size):
        fitness = fitness_function(tuple(arguments))
        if fitness > best_fitness:
            best_fitness = fitness
            best_arguments = tuple(arguments)

    return best_arguments


def make_lists(
        no_dimensions: int,
        step_size: Union[Tuple[float], float],
        centre_steps=True
):
    """
        Returns a list of lists of floats covering every combination across no_dimensions of points of integer step size
    between 0 and 1 inclusive.

    Parameters
    ----------
    no_dimensions
        The number of dimensions, that is the length of the lists
    step_size
        The step size. This can be a float or a tuple with the same number of dimensions
    centre_steps

    Returns
    -------
    lists: [[float]]
        A list of lists
    """
    if isinstance(step_size, float):
        step_size = tuple(
            step_size
            for _
            in range(no_dimensions)
        )

    if no_dimensions == 0:
        return [[]]

    sub_lists = make_lists(
        no_dimensions - 1,
        step_size[1:],
        centre_steps=centre_steps
    )
    step_size = step_size[0]
    return [
        [
            step_size * value + (
                0.5 * step_size
                if centre_steps
                else 0)
        ] + sub_list
        for value in range(int((1 / step_size)))
        for sub_list in sub_lists
    ]
