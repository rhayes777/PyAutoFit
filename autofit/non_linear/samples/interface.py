from abc import ABC, abstractmethod
from functools import wraps
from typing import Union, List, Tuple

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
