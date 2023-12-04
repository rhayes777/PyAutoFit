from abc import ABC
import csv
from functools import wraps
import json
import warnings
from copy import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from autofit import exc
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples.sample import Sample

from .summary import SamplesSummary
from .interface import SamplesInterface, to_instance
from ...text.formatter import write_table


### TODO: Rich how do I reduce this to one wrapper sensible?


def to_instance_sigma(func):
    """

    Parameters
    ----------
    func

    Returns
    -------
        A function that returns a 2D image.
    """

    @wraps(func)
    def wrapper(
        obj, sigma, as_instance: bool = True, *args, **kwargs
    ) -> Union[List, ModelInstance]:
        """
        This decorator checks if a light profile is a `LightProfileOperated` class and therefore already has had operations like a
        PSF convolution performed.

        This is compared to the `only_operated` input to determine if the image of that light profile is returned, or
        an array of zeros.

        Parameters
        ----------
        obj
            A light profile with an `image_2d_from` function whose class is inspected to determine if the image is
            operated on.
        grid
            A grid_like object of (y,x) coordinates on which the function values are evaluated.
        operated_only
            By default this is None and the image is returned irrespecive of light profile class (E.g. it does not matter
            if it is already operated or not). If this input is included as a bool, the light profile image is only
            returned if they are or are not already operated.

        Returns
        -------
            The 2D image, which is customized depending on whether it has been operated on.
        """

        vector = func(obj, sigma, as_instance, *args, **kwargs)

        if as_instance:
            return obj.model.instance_from_vector(
                vector=vector, ignore_prior_limits=True
            )

        return vector

    return wrapper


def to_instance_samples(func):
    """

    Parameters
    ----------
    func

    Returns
    -------
        A function that returns a 2D image.
    """

    @wraps(func)
    def wrapper(
        obj, sample_index, as_instance: bool = True, *args, **kwargs
    ) -> Union[List, ModelInstance]:
        """
        This decorator checks if a light profile is a `LightProfileOperated` class and therefore already has had operations like a
        PSF convolution performed.

        This is compared to the `only_operated` input to determine if the image of that light profile is returned, or
        an array of zeros.

        Parameters
        ----------
        obj
            A light profile with an `image_2d_from` function whose class is inspected to determine if the image is
            operated on.
        grid
            A grid_like object of (y,x) coordinates on which the function values are evaluated.
        operated_only
            By default this is None and the image is returned irrespecive of light profile class (E.g. it does not matter
            if it is already operated or not). If this input is included as a bool, the light profile image is only
            returned if they are or are not already operated.

        Returns
        -------
            The 2D image, which is customized depending on whether it has been operated on.
        """

        vector = func(obj, sample_index, as_instance, *args, **kwargs)

        if as_instance:
            return obj.model.instance_from_vector(
                vector=vector, ignore_prior_limits=True
            )

        return vector

    return wrapper


def to_instance_input(func):
    """

    Parameters
    ----------
    func

    Returns
    -------
        A function that returns a 2D image.
    """

    @wraps(func)
    def wrapper(
        obj, input_vector, as_instance: bool = True, *args, **kwargs
    ) -> Union[List, ModelInstance]:
        """
        This decorator checks if a light profile is a `LightProfileOperated` class and therefore already has had operations like a
        PSF convolution performed.

        This is compared to the `only_operated` input to determine if the image of that light profile is returned, or
        an array of zeros.

        Parameters
        ----------
        obj
            A light profile with an `image_2d_from` function whose class is inspected to determine if the image is
            operated on.
        grid
            A grid_like object of (y,x) coordinates on which the function values are evaluated.
        operated_only
            By default this is None and the image is returned irrespecive of light profile class (E.g. it does not matter
            if it is already operated or not). If this input is included as a bool, the light profile image is only
            returned if they are or are not already operated.

        Returns
        -------
            The 2D image, which is customized depending on whether it has been operated on.
        """

        vector = func(obj, input_vector, as_instance, *args, **kwargs)

        if as_instance:
            return obj.model.instance_from_vector(
                vector=vector, ignore_prior_limits=True
            )

        return vector

    return wrapper


class Samples(SamplesInterface, ABC):
    def __init__(
        self,
        model: AbstractPriorModel,
        sample_list: List[Sample],
        samples_info: Optional[Dict] = None,
        search_internal: Optional = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the search_internal of
        a `NonLinearSearch` (e.g. as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        This class stores samples of searches which provide maximum likelihood estimates of the  model-fit (e.g.
        PySwarms, LBFGS).

        To use a library's in-built visualization tools results are optionally stored in their native internal format
        using the `search_internal` attribute.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        sample_list
            The list of `Samples` which contains the paramoeters, likelihood, weights, etc. of every sample taken
            by the non-linear search.
        samples_info
            Contains information on the samples (e.g. total iterations, time to run the search, etc.).
        search_internal
            The nested sampler's results in their native internal format for interfacing its visualization library.
        """

        super().__init__(model=model)

        self.sample_list = sample_list
        self.samples_info = samples_info
        self.search_internal = search_internal

    @property
    def log_evidence(self):
        return None

    @classmethod
    def from_csv(cls, paths, model: AbstractPriorModel):
        """
        Returns a `Samples` object from the output paths of a non-linear search.

        This function loads the sample values (e.g. parameters, log likelihoods) from a .csv file, which is a
        standardized output for all **PyAutoFit** non-linear searches.

        The samples object requires additional information on the non-linear search (e.g. the number of live points),
        which is loaded from the `samples_info.json` file.

        This function also looks for the internal results of the non-linear search and includes them in the samples if
        they exists, which allows for the search's internal visualization and analysis tools to be used.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        model
            An object that represents possible instances of some model with a given dimensionality which is the number
            of free dimensions of the model.

        Returns
        -------
        The samples which have been loaded from hard-disk via .csv.
        """

        sample_list = paths.load_samples()
        samples_info = paths.load_samples_info()

        try:
            search_internal = paths.load_search_internal()
        except FileNotFoundError:
            search_internal = None

        return cls.from_list_info_and_model(
            sample_list=sample_list,
            samples_info=samples_info,
            model=model,
            search_internal=search_internal,
        )

    @classmethod
    def from_list_info_and_model(
        cls,
        sample_list,
        samples_info,
        model: AbstractPriorModel,
        search_internal=None,
    ):
        try:
            auto_correlation_settings = AutoCorrelationsSettings(
                check_for_convergence=True,
                check_size=samples_info["check_size"],
                required_length=samples_info["required_length"],
                change_threshold=samples_info["change_threshold"],
            )
        except (KeyError, NameError):
            auto_correlation_settings = None

        try:
            return cls(
                model=model,
                sample_list=sample_list,
                samples_info=samples_info,
                search_internal=search_internal,
                auto_correlation_settings=auto_correlation_settings,
            )
        except TypeError:
            return cls(
                model=model,
                sample_list=sample_list,
                samples_info=samples_info,
                search_internal=search_internal,
            )

    def summary(self):
        return SamplesSummary(
            max_log_likelihood_sample=self.max_log_likelihood_sample,
            model=self.model,
        )

    def __add__(self, other: "Samples") -> "Samples":
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        Parameters
        ----------
        other
            The Samples to be added to this Samples instance.

        Returns
        -------
        A class that combined the samples of the two Samples objects.
        """

        self._check_addition(other=other)

        if self.search_internal is not None:
            warnings.warn(
                f"Addition of {self.__class__.__name__} cannot retain results in native format. "
                "Visualization of summed samples diabled.",
                exc.SamplesWarning,
            )

        return self.__class__(
            model=self.model,
            sample_list=self.sample_list + other.sample_list,
        )

    def __radd__(self, other):
        """
        Samples can be added together, which combines their `sample_list` meaning that inferred parameters are
        computed via their joint PDF.

        Overwriting `__radd__` enables the sum function to be used on a list of samples, e.g.:

        `samples = sum([samples_x5, samples_x5, samples_x5])`

        Parameters
        ----------
        other
            The Samples to be added to this Samples instance.

        Returns
        -------
        A class that combines the samples of a list of Samples objects.
        """
        return self

    def __len__(self):
        return len(self.sample_list)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._names = None
        result._paths = None
        return result

    def _check_addition(self, other: "Samples"):
        """
        When adding samples together, perform the following checks to make sure it is valid to add the two objects
        together:

        - That both objects being added are `Samples` objects.
        - That both models have the same prior count, else the dimensionality does not allow for valid addition.
        - That both `Samples` objects use an identical model, such that we are adding together the same parameters.

        Parameters
        ----------
        other
            The Samples to be added to this Samples instance.
        """

        def raise_exc():
            raise exc.SamplesException(
                "Cannot add together two Samples objects which have different models."
            )

        if not isinstance(self, Samples):
            raise_exc()

        if not isinstance(other, Samples):
            raise_exc()

        if self.model.prior_count != other.model.prior_count:
            raise_exc()

        for path_self, path_other in zip(self.model.paths, other.model.paths):
            if path_self != path_other:
                raise_exc()

    def values_for_path(self, path: Tuple[str]) -> List[float]:
        """
        Returns the value for a variable with a given path
        for each sample in the model
        """
        return [sample.kwargs[path] for sample in self.sample_list]

    @property
    def total_iterations(self) -> int:
        return self.samples_info["total_iterations"]

    @property
    def time(self) -> Optional[float]:
        return self.samples_info["time"]

    @property
    def parameter_lists(self):
        result = list()
        for sample in self.sample_list:
            tuples = self.paths if sample.is_path_kwargs else self.names
            result.append(sample.parameter_lists_for_paths(tuples))

        return result

    @property
    def total_samples(self):
        return len(self.sample_list)

    @property
    def weight_list(self):
        return [sample.weight for sample in self.sample_list]

    @property
    def log_likelihood_list(self):
        return [sample.log_likelihood for sample in self.sample_list]

    @property
    def log_posterior_list(self):
        return [sample.log_posterior for sample in self.sample_list]

    @property
    def log_prior_list(self):
        return [sample.log_prior for sample in self.sample_list]

    @property
    def parameters_extract(self):
        return np.asarray(self.parameter_lists).T

    @property
    def _headers(self) -> List[str]:
        """
        Headers for the samples table
        """

        return self.model.joined_paths + [
            "log_likelihood",
            "log_prior",
            "log_posterior",
            "weight",
        ]

    @property
    def _rows(self) -> List[List[float]]:
        """
        Rows in the samples table
        """

        log_likelihood_list = self.log_likelihood_list
        log_prior_list = self.log_prior_list
        log_posterior_list = self.log_posterior_list
        weight_list = self.weight_list

        for index, row in enumerate(self.parameter_lists):
            yield row + [
                log_likelihood_list[index],
                log_prior_list[index],
                log_posterior_list[index],
                weight_list[index],
            ]

    def write_table(self, filename: str):
        """
        Write a table of parameters, posteriors, priors and likelihoods.

        Parameters
        ----------
        filename
            Where the table is to be written
        """
        write_table(
            filename=filename,
            headers=list(self._headers),
            rows=list(self._rows),
        )

    def info_to_json(self, filename):
        with open(filename, "w") as outfile:
            json.dump(self.samples_info, outfile)

    @property
    def max_log_likelihood_sample(self) -> Sample:
        """
        The index of the sample with the highest log likelihood.
        """
        most_likely_sample = None
        for sample in self.sample_list:
            if (
                most_likely_sample is None
                or sample.log_likelihood > most_likely_sample.log_likelihood
            ):
                most_likely_sample = sample
        return most_likely_sample

    @to_instance
    def max_log_likelihood(self, as_instance: bool = True) -> List[float]:
        """
        The parameters of the maximum log likelihood sample of the `NonLinearSearch` returned as a model instance or
        list of values.
        """

        sample = self.max_log_likelihood_sample

        return sample.parameter_lists_for_paths(
            self.paths if sample.is_path_kwargs else self.names
        )

    @property
    def max_log_posterior_sample(self) -> Sample:
        return self.sample_list[self.max_log_posterior_index]

    @property
    def max_log_posterior_index(self) -> int:
        """
        The index of the sample with the highest log posterior.
        """
        return int(np.argmax(self.log_posterior_list))

    @to_instance
    def max_log_posterior(self, as_instance: bool = True) -> ModelInstance:
        """
        The parameters of the maximum log posterior sample of the `NonLinearSearch` returned as a model instance.
        """
        return self.parameter_lists[self.max_log_posterior_index]

    @to_instance_samples
    def from_sample_index(
        self, sample_index: int, as_instance: bool = True
    ) -> ModelInstance:
        """
        The parameters of an individual sample of the non-linear search, returned as a model instance.

        Parameters
        ----------
        sample_index
            The sample index of the weighted sample to return.
        """
        return self.parameter_lists[sample_index]

    def minimise(self) -> "Samples":
        """
        A copy of this object with only important samples retained
        """
        samples = copy(self)
        samples.model = None
        samples.sample_list = list(
            {self.max_log_likelihood_sample, self.max_log_posterior_sample}
        )
        return samples

    def with_paths(self, paths: Union[List[Tuple[str, ...]], List[str]]) -> "Samples":
        """
        Create a copy of this object with only attributes specified
        by a list of paths.

        Parameters
        ----------
        paths
            A list of paths to attributes. Only kwargs and model components
            specified by these paths are retained.

            All children of a given path are retained.

        Returns
        -------
        A set of samples with a reduced set of attributes
        """
        with_paths = copy(self)
        with_paths.model = self.model.with_paths(paths)
        with_paths.sample_list = [
            sample.with_paths(paths) for sample in self.sample_list
        ]
        return with_paths

    def without_paths(
        self, paths: Union[List[Tuple[str, ...]], List[str]]
    ) -> "Samples":
        """
        Create a copy of this object with only attributes not specified
        by a list of paths.

        Parameters
        ----------
        paths
            A list of paths to attributes. kwargs and model components
            specified by these paths are removed.

            All children of a given path are removed.

        Returns
        -------
        A set of samples with a reduced set of attributes
        """
        with_paths = copy(self)
        with_paths.model = self.model.without_paths(paths)
        with_paths.sample_list = [
            sample.without_paths(paths) for sample in self.sample_list
        ]
        return with_paths

    def subsamples(self, model):
        if self.model is None:
            return None

        path_map = {
            tuple(self.model.all_paths_for_prior(prior)): path
            for path, prior in model.path_priors_tuples
        }
        copied = copy(self)
        copied._paths = None
        copied._names = None
        copied.model = model

        copied.sample_list = [sample.subsample(path_map) for sample in self.sample_list]
        return copied
