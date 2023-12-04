from abc import ABC, abstractmethod
from functools import wraps
from typing import Union, List, Tuple

from autofit import conf
from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.abstract import Path


def to_instance(func):
    @wraps(func)
    def wrapper(
        obj, as_instance: bool = True, *args, **kwargs
    ) -> Union[List, ModelInstance]:
        vector = func(obj, as_instance, *args, **kwargs)

        if as_instance:
            return obj.model.instance_from_vector(
                vector=vector, ignore_prior_limits=True
            )

        return vector

    return wrapper


class SamplesInterface(ABC):
    def __init__(self, model: AbstractPriorModel):
        """
        An interface for the samples of a `NonLinearSearch` that has been run, including the maximum log likelihood
        """
        self.model = model

        self._paths = None
        self._names = None
        self._instance = None

    @property
    def instance(self):
        if self._instance is None:
            self._instance = self.max_log_likelihood()
        return self._instance

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
    @abstractmethod
    def max_log_likelihood_sample(self):
        pass

    @property
    def log_likelihood(self):
        return self.max_log_likelihood_sample.log_likelihood

    @property
    @abstractmethod
    def log_evidence(self):
        pass

    def model_absolute(self, a: float) -> AbstractPriorModel:
        """
        Returns a model where every free parameter is a `GaussianPrior` with `mean` the previous result's
        inferred maximum log likelihood parameter values and `sigma` the input absolute value `a`.

        For example, a previous result may infer a parameter to have a maximum log likelihood value of 2.

        If this result is used for search chaining, `model_absolute(a=0.1)` will assign this free parameter
        `GaussianPrior(mean=2.0, sigma=0.1)` in the new model, where `sigma` is linked to the input `a`.

        Parameters
        ----------
        a
            The absolute width of gaussian priors

        Returns
        -------
        A model mapper created by taking results from this search and creating priors with the defined absolute
        width.
        """
        return self.model.mapper_from_gaussian_tuples(
            self.gaussian_priors_at_sigma(sigma=self.sigma), a=a
        )

    def model_relative(self, r: float) -> AbstractPriorModel:
        """
        Returns a model where every free parameter is a `GaussianPrior` with `mean` the previous result's
        inferred maximum log likelihood parameter values and `sigma` a relative value from the result `r`.

        For example, a previous result may infer a parameter to have a maximum log likelihood value of 2 and
        an error at the input `sigma` of 0.5.

        If this result is used for search chaining, `model_relative(r=0.1)` will assign this free parameter
        `GaussianPrior(mean=2.0, sigma=0.5*0.1)` in the new model, where `sigma` is the inferred error times `r`.

        Parameters
        ----------
        r
            The relative width of gaussian priors

        Returns
        -------
        A model mapper created by taking results from this search and creating priors with the defined relative
        width.
        """
        return self.model.mapper_from_gaussian_tuples(
            self.gaussian_priors_at_sigma(sigma=self.sigma), r=r
        )

    def model_bounded(self, b: float) -> AbstractPriorModel:
        """
        Returns a model where every free parameter is a `UniformPrior` with `lower_limit` and `upper_limit the previous
        result's inferred maximum log likelihood parameter values minus and plus the bound `b`.

        For example, a previous result may infer a parameter to have a maximum log likelihood value of 2.

        If this result is used for search chaining, `model_bound(b=0.1)` will assign this free parameter
        `UniformPrior(lower_limit=1.9, upper_limit=2.1)` in the new model.

        Parameters
        ----------
        b
            The size of the bounds of the uniform prior

        Returns
        -------
        A model mapper created by taking results from this search and creating priors with the defined bounded
        uniform priors.
        """
        return self.model.mapper_from_uniform_floats(
            floats=self.max_log_likelihood(as_instance=False), b=b
        )

    @property
    def sigma(self):
        return conf.instance["general"]["prior_passer"]["sigma"]

    def gaussian_priors_at_sigma(self, sigma: float) -> [List]:
        """
        Returns `GaussianPrior`'s of every parameter in a fit for use with non-linear search chaining.

        `GaussianPrior`s of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent search, where:

        - The mean is given by maximum log likelihood model values.
        - Their errors are omitted, as this information is not available from an search. When these priors are
          used to link to another search, it will thus automatically use the prior config values.

        Parameters
        ----------
        sigma
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        return list(
            map(
                lambda vector: (vector, 0.0), self.max_log_likelihood(as_instance=False)
            )
        )
