import csv
from copy import copy
from typing import List, Tuple

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.tools.util import split_paths


class Sample:
    def __init__(
            self,
            log_likelihood: float,
            log_prior: float,
            weight: float,
            kwargs=None
    ):
        """
        One sample taken during a search

        Parameters
        ----------
        log_likelihood
            The likelihood associated with this instance
        log_prior
            A logarithmic prior of the instance
        weight
            The weight this sample contributes to the PDF.
        kwargs
            Dictionary mapping model paths to values for the sample
        """
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.weight = weight
        self.kwargs = kwargs or dict()

    @property
    def log_posterior(self) -> float:
        """
        Compute the posterior
        """
        return self.log_likelihood + self.log_prior

    def parameter_lists_for_model(
            self,
            model: AbstractPriorModel
    ) -> List[float]:
        """
        Values for instantiating a model, in the same order as priors
        from the model.

        Parameters
        ----------
        model
            The model from which this was a sample

        Returns
        -------
        A list of physical values
        """
        if self.is_path_kwargs:
            paths = model.all_paths
        else:
            paths = model.all_names

        return self.parameter_lists_for_paths(
            paths
        )

    def parameter_lists_for_paths(
            self,
            paths
    ):
        result = list()
        for keys in paths:
            is_found = False
            for key in keys:
                if key in self.kwargs:
                    result.append(
                        self.kwargs[
                            key
                        ]
                    )
                    is_found = True
                    break
            if not is_found:
                raise KeyError(
                    f"Could not find any of the following keys in kwargs {keys}"
                )
        return result

    @property
    def is_path_kwargs(self) -> bool:
        """
        Are the keys in the kwargs dictionary tuples? If they
        are this indicates that they are explicit paths through
        the model.
        """
        for key in self.kwargs.keys():
            return isinstance(
                key, tuple
            )
        return False

    def subsample(self, path_map):
        arg_dict = {}
        for paths, new_path in path_map.items():
            for path in paths:
                if path in self.kwargs:
                    arg_dict[new_path] = self.kwargs[path]
                    break
            else:
                raise KeyError(f"No path from {paths} in sample")

        return type(self)(
            log_likelihood=self.log_likelihood,
            log_prior=self.log_prior,
            weight=self.weight,
            kwargs=arg_dict
        )

    @classmethod
    def from_lists(
            cls,
            model: AbstractPriorModel,
            parameter_lists: List[List[float]],
            log_likelihood_list: List[float],
            log_prior_list: List[float],
            weight_list: List[float]
    ) -> List["Sample"]:
        """
        Convenience method to create a list of samples from lists of contained values
        """
        samples = list()

        paths = model.unique_prior_paths

        for params, log_likelihood, log_prior, weight in zip(
                parameter_lists,
                log_likelihood_list,
                log_prior_list,
                weight_list
        ):
            arg_dict = {
                t: param
                for t, param
                in zip(
                    paths,
                    params
                )
            }

            samples.append(
                cls(
                    log_likelihood=log_likelihood,
                    log_prior=log_prior,
                    weight=weight,
                    kwargs=arg_dict
                )
            )
        return samples

    def instance_for_model(self, model: AbstractPriorModel):
        """
        Create an instance from this sample for a model

        Parameters
        ----------
        model
            The model the this sample was taken from

        Returns
        -------
        The instance corresponding to this sample
        """
        try:
            if self.is_path_kwargs:
                return model.instance_from_path_arguments(
                    self.kwargs
                )
            else:
                return model.instance_from_prior_name_arguments(
                    self.kwargs
                )

        except KeyError:
            # TODO: Does this get used? If so, why?
            return model.instance_from_vector(
                self.parameter_lists_for_model(model)
            )

    @split_paths
    def with_paths(
            self,
            paths: List[Tuple[str, ...]]
    ) -> "Sample":
        """
        Create a copy of this object retaining only the kwargs for which
        there is a matching path in paths.

        Parameters
        ----------
        paths
            A list of paths for which attributes should be retained.

        Returns
        -------
        A reduced sample
        """
        with_paths = copy(self)
        with_paths.kwargs = {
            key: value
            for key, value
            in self.kwargs.items()
            if any(
                all(
                    first == second
                    for first, second
                    in zip(
                        key, path
                    )
                )
                for path in paths
            )
        }
        return with_paths

    @split_paths
    def without_paths(
            self,
            paths: List[Tuple[str, ...]]
    ) -> "Sample":
        """
        Create a copy of this object retaining only the kwargs for which
        there is no matching path in paths.

        Parameters
        ----------
        paths
            A list of paths for which attributes should be removed.

        Returns
        -------
        A reduced sample
        """
        without_paths = copy(self)
        without_paths.kwargs = {
            key: value
            for key, value
            in self.kwargs.items()
            if not any(
                all(
                    first == second
                    for first, second
                    in zip(
                        key, path
                    )
                )
                for path in paths
            )
        }
        return without_paths


def load_from_table(filename: str) -> List[Sample]:
    """
    Load samples from a table

    Parameters
    ----------
    filename
        The path to a CSV file

    Returns
    -------
    A list of samples, one for each row in the CSV
    """
    samples = list()

    sample_args = (
        "log_likelihood",
        "log_prior",
        "weight",
    )

    with open(filename, "r+", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            d = {
                header: float(value)
                for header, value
                in zip(
                    headers,
                    row
                )
            }

            samples.append(
                Sample(
                    **{
                        key: value
                        for key, value
                        in d.items()
                        if key in sample_args
                    },
                    kwargs={
                        key: value
                        for key, value
                        in d.items()
                        if key not in sample_args
                    }
                )
            )

    return samples
