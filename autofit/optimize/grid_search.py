import os

import numpy as np

from autofit import conf
from autofit import exc
from autofit.mapper import link
from autofit.mapper import model_mapper as mm
from autofit.mapper import prior as p
from autofit.optimize import non_linear
from autofit.optimize import optimizer


class GridSearchResult(object):
    def __init__(self, results, lists):
        """
        The result of a grid search.

        Parameters
        ----------
        results: [non_linear.Result]
            The results of the non linear optimizations performed at each grid step
        lists: [[float]]
            A list of lists of values representing the lower bounds of the grid searched values at each step
        """
        self.lists = lists
        self.results = results
        self.no_dimensions = len(self.lists[0])
        self.no_steps = len(self.lists)
        self.side_length = int(self.no_steps ** (1 / self.no_dimensions))

    @property
    def figure_of_merit_array(self):
        """
        Returns
        -------
        figure_of_merit_array: np.ndarray
            An array of figures of merit. This array has the same dimensionality as the grid search, with the value in
            each entry being the figure of merit taken from the optimization performed at that point.
        """
        return np.reshape(np.array([result.figure_of_merit for result in self.results]),
                          tuple(self.side_length for _ in range(self.no_dimensions)))


class GridSearch(object):
    def __init__(self, number_of_steps=10, optimizer_class=non_linear.DownhillSimplex, model_mapper=None,
                 name="grid_search"):
        """
        Performs a non linear optimiser search for each square in a grid. The dimensionality of the search depends on
        the number of distinct priors passed to the fit function. (1 / step_size) ^ no_dimension steps are performed
        per an optimisation.

        Parameters
        ----------
        number_of_steps: int
            The number of steps to go in each direction
        optimizer_class: class
            The class of the optimizer that is run at each step
        model_mapper: mm.ModelMapper | None
            The model mapper that maps between the optimizer and class model
        name: str
            The name of this grid search
        """
        self.variable = model_mapper or mm.ModelMapper()
        self.name = name
        self.number_of_steps = number_of_steps
        self.optimizer_class = optimizer_class

        sym_path = "{}/{}/optimizer".format(conf.instance.output_path, name)
        self.backup_path = "{}/{}/optimizer_backup".format(conf.instance.output_path, name)

        try:
            os.makedirs("/".join(sym_path.split("/")[:-1]))
        except FileExistsError:
            pass

        self.path = link.make_linked_folder(sym_path)

    @property
    def hyper_step_size(self):
        """
        Returns
        -------
        hyper_step_size: float
            The size of a step in any given dimension in hyper space.
        """
        return 1 / self.number_of_steps

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
        return optimizer.make_lists(len(grid_priors), step_size=self.hyper_step_size, include_upper_limit=False)

    def models_mappers(self, grid_priors):
        """
        A generator that yields one model mapper at a time. Each model mapper represents on step in the grid search. Any
        prior that is not included in grid priors remains unchanged; priors included in grid priors are replaced by
        uniform priors between the limits of the grid step:

        UniformPrior(lower_limit=lower_limit + value * prior_step_size,
                     upper_limit=lower_limit + (value + self.step_size) * prior_step_size)

        Parameters
        ----------
        grid_priors: [p.Prior]
            A list of priors to be substituted for uniform priors across the grid.

        Returns
        -------
        model_mappers: generator[mm.ModelMapper]
        """
        grid_priors = set(grid_priors)
        lists = self.make_lists(grid_priors)
        for values in lists:
            arguments = {}
            for value, grid_prior in zip(values, grid_priors):
                prior_step_size = grid_prior.upper_limit - grid_prior.lower_limit
                if float("-inf") == grid_prior.lower_limit or float('inf') == grid_prior.upper_limit:
                    raise exc.PriorException("Priors passed to the grid search must have definite limits")
                lower_limit = grid_prior.lower_limit + value * prior_step_size
                upper_limit = grid_prior.lower_limit + (value + self.hyper_step_size) * prior_step_size
                prior = p.UniformPrior(lower_limit=lower_limit, upper_limit=upper_limit)
                arguments[grid_prior] = prior
            yield self.variable.mapper_from_partial_prior_arguments(arguments)

    def fit(self, analysis, grid_priors):
        """
        Fit an analysis with a set of grid priors. The grid priors are priors associated with the model mapper
        of this instance that are replaced by uniform priors for each step of the grid search.

        Parameters
        ----------
        analysis: non_linear.Analysis
            An analysis used to determine the fitness of a given model instance
        grid_priors: [p.Prior]
            A list of priors to be substituted for uniform priors across the grid.

        Returns
        -------
        result: GridSearchResult
            An object that comprises the results from each individual fit
        """
        results = []
        lists = self.make_lists(grid_priors)
        for values, model_mapper in zip(lists, self.models_mappers(grid_priors)):
            result = self.optimizer_class(model_mapper=model_mapper,
                                          name="{}/{}".format(self.name, "_".join(map(str, values)))).fit(analysis)
            results.append(result)
        return GridSearchResult(results, lists)
