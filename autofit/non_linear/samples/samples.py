from abc import ABC
import csv
from functools import wraps
import json
import warnings
from copy import copy
from typing import List, Optional, Tuple, Union

import numpy as np

from autofit import exc
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel, Path
from autofit.non_linear.samples.sample import Sample


### TODO: Rich how do I reduce this to one wrapper sensible?

def to_instance(func):
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
        obj,
        as_instance: bool = True,
        *args,
        **kwargs
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

        vector = func(obj, as_instance, *args, **kwargs)

        if as_instance:

            return obj.model.instance_from_vector(
                vector=vector,
                ignore_prior_limits=True
            )

        return vector

    return wrapper


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
        obj,
        sigma,
        as_instance : bool = True,
        *args,
        **kwargs
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
                vector=vector,
                ignore_prior_limits=True
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
        obj,
        sample_index,
        as_instance : bool = True,
        *args,
        **kwargs
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
                vector=vector,
                ignore_prior_limits=True
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
        obj,
        input_vector,
        as_instance : bool = True,
        *args,
        **kwargs
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
                vector=vector,
                ignore_prior_limits=True
            )

        return vector

    return wrapper


class Samples(ABC):
    def __init__(
            self,
            model: AbstractPriorModel,
            sample_list: List[Sample],
            total_iterations: Optional[int] = None,
            time: Optional[float] = None,
            results_internal: Optional = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the results_internal of
        a `NonLinearSearch` (e.g. as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        This class stores samples of searches which provide maximum likelihood estimates of the  model-fit (e.g.
        PySwarms, LBFGS).

        To use a library's in-built visualization tools results are optionally stored in their native internal format
        using the `results_internal` attribute.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        sample_list
            The list of `Samples` which contains the paramoeters, likelihood, weights, etc. of every sample taken
            by the non-linear search.
        total_iterations
            The total number of iterations, which often cannot be estimated from the sample list (which contains
            only accepted samples).
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        results_internal
            The nested sampler's results in their native internal format for interfacing its visualization library.
        """
        self.model = model
        self.sample_list = sample_list

        self.total_iterations = total_iterations
        self.time = time
        self.results_internal = results_internal

        self._paths = None
        self._names = None

    def __add__(
            self,
            other: "Samples"
    ) -> "Samples":
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

        if self.results_internal is not None:
            warnings.warn(
                f"Addition of {self.__class__.__name__} cannot retain results in native format. "
                "Visualization of summed samples diabled.",
                exc.SamplesWarning
            )

        return self.__class__(
            model=self.model,
            sample_list=self.sample_list + other.sample_list,
            time=self.time
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

        if not isinstance(
                self,
                Samples
        ):

            raise_exc()

        if not isinstance(
                other,
                Samples
        ):
            raise_exc()

        if self.model.prior_count != other.model.prior_count:
            raise_exc()

        for path_self, path_other in zip(self.model.paths, other.model.paths):
            if path_self != path_other:
                raise_exc()

    def values_for_path(
            self,
            path: Tuple[str]
    ) -> List[float]:
        """
        Returns the value for a variable with a given path
        for each sample in the model
        """
        return [
            sample.kwargs[
                path
            ]
            for sample
            in self.sample_list
        ]

    @property
    def paths(self) -> List[Tuple[Path]]:
        """
        A list of paths to unique priors in the same order as prior
        ids (and therefore sample columns)

        Uses hasattr to make backwards compatible
        """
        if not hasattr(self, "_paths") or self._paths is None:
            self._paths = self.model.all_paths
        return self._paths

    @property
    def names(self) -> List[Tuple[str]]:
        """
        A list of names of unique priors in the same order as prior
        ids (and therefore sample columns)

        Uses hasattr to make backwards compatible
        """
        if not hasattr(self, "_names") or self._names is None:
            self._names = self.model.all_names
        return self._names

    @property
    def parameter_lists(self):
        result = list()
        for sample in self.sample_list:
            tuples = self.paths if sample.is_path_kwargs else self.names
            result.append(
                sample.parameter_lists_for_paths(
                    tuples
                )
            )

        return result

    @property
    def total_samples(self):
        return len(self.sample_list)

    @property
    def weight_list(self):
        return [
            sample.weight
            for sample
            in self.sample_list
        ]

    @property
    def log_likelihood_list(self):
        return [
            sample.log_likelihood
            for sample
            in self.sample_list
        ]

    @property
    def log_posterior_list(self):
        return [
            sample.log_posterior
            for sample
            in self.sample_list
        ]

    @property
    def log_prior_list(self):
        return [
            sample.log_prior
            for sample
            in self.sample_list
        ]

    @property
    def parameters_extract(self):
        return np.asarray(self.parameter_lists).T

    @property
    def _headers(self) -> List[str]:
        """
        Headers for the samples table
        """

        return self.model.model_component_and_parameter_names + [
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

        with open(filename, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._headers)
            for row in self._rows:
                writer.writerow(row)

    @property
    def info_json(self):
        return {}

    def info_to_json(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.info_json, outfile)

    @property
    def max_log_likelihood_sample(self) -> Sample:
        """
        The index of the sample with the highest log likelihood.
        """
        most_likely_sample = None
        for sample in self.sample_list:
            if most_likely_sample is None or sample.log_likelihood > most_likely_sample.log_likelihood:
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
        return self.sample_list[
            self.max_log_posterior_index
        ]

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
    def from_sample_index(self, sample_index : int, as_instance: bool = True) -> ModelInstance:
        """
        The parameters of an individual sample of the non-linear search, returned as a model instance.

        Parameters
        -----------
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
        samples.sample_list = list({
            self.max_log_likelihood_sample,
            self.max_log_posterior_sample
        })
        return samples

    def with_paths(
            self,
            paths: Union[List[Tuple[str, ...]], List[str]]
    ) -> "Samples":
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
        with_paths.model = self.model.with_paths(
            paths
        )
        with_paths.sample_list = [
            sample.with_paths(paths)
            for sample in self.sample_list
        ]
        return with_paths

    def without_paths(
            self,
            paths: Union[List[Tuple[str, ...]], List[str]]
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
        with_paths.model = self.model.without_paths(
            paths
        )
        with_paths.sample_list = [
            sample.without_paths(paths)
            for sample in self.sample_list
        ]
        return with_paths

    def subsamples(self, model):
        if self.model is None:
            return None

        path_map = {
            tuple(self.model.all_paths_for_prior(
                prior
            )): path
            for path, prior in model.path_priors_tuples
        }
        copied = copy(self)
        copied._paths = None
        copied._names = None
        copied.model = model

        copied.sample_list = [
            sample.subsample(path_map)
            for sample in self.sample_list
        ]
        return copied

    def gaussian_priors_at_sigma(self, sigma: float) -> [List]:
        """
        `GaussianPrior`s of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent search, where:

        - The mean is given by maximum log likelihood model values.
        - Their errors are omitted, as this information is not available from an search. When these priors are
        used to link to another search, it will thus automatically use the prior config values.

        Parameters
        -----------
        sigma
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        return list(map(lambda vector: (vector, 0.0), self.max_log_likelihood(as_instance=False)))