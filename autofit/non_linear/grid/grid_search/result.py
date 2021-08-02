from typing import List, Tuple

import numpy as np

from autofit import exc
from autofit.mapper import model_mapper as mm
from autofit.non_linear.result import Result


class GridSearchResult:

    def __init__(
            self,
            shape_native: Tuple[int],
            results: List[Result],
            lower_limit_lists: List[List[float]],
            physical_lower_limits_lists: List[List[float]],
    ):
        """
        The result of a grid search.

        The results are stored as a list of values or lists, which does not account for the dimensionality of the
        grid search. For example, the results are stored as a 1D list irrespective of whether the grid search was
        performed over one, two or more dimensions.

        The `native` methods map the results from these 1D list structures to the higher dimensional grid dimensions.

        Parameters
        ----------
        results
            The results of the non linear optimizations performed at each grid cell, stored as a list of results.
        lower_limit_lists
            The lower bounds of the grid search unit prior values for each cell of the grid search. This is stored
            as a list of lists, where the outer list contains an entry for every grid cell and inner list has the lower
            bound of every unit prior.
        physical_lower_limits_lists
            The lower bounds of the grid search physical prior values for each cell of the grid search. This is stored
            as a list of lists, where the outer list contains an entry for every grid cell and inner list has the lower
            bound of every physical prior.
        """
        self.results = results
        self.lower_limit_lists = lower_limit_lists
        self.physical_lower_limits_lists = physical_lower_limits_lists
        self.shape_native = shape_native

    def __getattr__(self, item: str) -> object:
        """
        We default to getting attributes from the best result. This allows promises to reference best results.
        """
        return getattr(self.best_result, item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def no_dimensions(self):
        """
        The number of dimensions of the grid search, where a dimension corresponds to a specific parameter in a model.
        """
        return len(self.shape_native)

    @property
    def no_steps(self):
        """
        The number of steps taken in every dimension of the grid search.
        """
        return self.shape_native[0]

    @property
    def side_length(self):
        return int(self.no_steps ** (1 / self.no_dimensions))

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
    def physical_step_sizes(self) -> Tuple[int]:
        """
        The largest physical step sizes of every parameter

        Returns
        -------

        """

        physical_step_sizes = []

        # TODO : Make this work for all dimensions in a less ugly way.

        for dim in range(self.no_dimensions):

            values = [value[dim] for value in self.physical_lower_limits_lists]
            diff = [abs(values[n] - values[n - 1]) for n in range(1, len(values))]

            if dim == 0:
                physical_step_sizes.append(np.max(diff))
            elif dim == 1:
                physical_step_sizes.append(np.min(diff))
            else:
                raise exc.GridSearchException(
                    "This feature does not support > 2 dimensions"
                )

        return tuple(physical_step_sizes)

    @property
    def physical_centres_lists(self):
        return [
            [
                lower_limit[dim] + self.physical_step_sizes[dim] / 2
                for dim in range(self.no_dimensions)
            ]
            for lower_limit in self.physical_lower_limits_lists
        ]

    @property
    def physical_upper_limits_lists(self):
        return [
            [
                lower_limit[dim] + self.physical_step_sizes[dim]
                for dim in range(self.no_dimensions)
            ]
            for lower_limit in self.physical_lower_limits_lists
        ]

    @property
    def results_native(self):
        """
        The result of every grid search on a NumPy array whose shape is the native dimensions of the grid search.

        For example, for a 2x2 grid search the shape of the Numpy array is (2,2) and it is numerically ordered such
        that the first search's result (corresponding to unit priors (0.0, 0.0)) are in the first value (E.g. entry
        [0, 0]) of the NumPy array.
        """
        return np.reshape(
            np.array([result for result in self.results]),
            self.shape_native,
        )

    @property
    def log_likelihoods_native(self):
        """
        The maximum log likelihood of every grid search on a NumPy array whose shape is the native dimensions of the
        grid search.

        For example, for a 2x2 grid search the shape of the Numpy array is (2,2) and it is numerically ordered such
        that the first search's maximum likelihood (corresponding to unit priors (0.0, 0.0)) are in the first
        value (E.g. entry [0, 0]) of the NumPy array.
        """
        return np.reshape(
            np.array([result.log_likelihood for result in self.results]),
            self.shape_native
        )

    @property
    def log_evidences_native(self):
        """
        The log evidence of every grid search on a NumPy array whose shape is the native dimensions of the grid search.

        For example, for a 2x2 grid search the shape of the Numpy array is (2,2) and it is numerically ordered such
        that the first search's log evidence (corresponding to unit priors (0.0, 0.0)) are in the first value (E.g.
        entry [0, 0]) of the NumPy array.
        """
        return np.reshape(
            np.array([result.samples.log_evidence for result in self.results]),
            self.shape_native
        )