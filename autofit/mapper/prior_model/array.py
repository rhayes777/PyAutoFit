from typing import Tuple, Dict

from .abstract import AbstractPriorModel
from autofit.mapper.prior.abstract import Prior
import numpy as np


class Array(AbstractPriorModel):
    def __init__(self, shape: Tuple[int, ...], prior: Prior):
        """
        An array of priors.

        Parameters
        ----------
        shape : (int, int)
            The shape of the array.
        prior : Prior
            The prior of every entry in the array.
        """
        super().__init__()
        self.shape = shape
        self.indices = np.ndindex(*shape)

        for index in self.indices:
            setattr(
                self,
                self._make_key(index),
                prior.new(),
            )

    @staticmethod
    def _make_key(index):
        suffix = "_".join(map(str, index))
        return f"prior_{suffix}"

    def _instance_for_arguments(
        self,
        arguments: Dict[Prior, float],
        ignore_assertions: bool = False,
    ):
        array = np.zeros(self.shape)
        for index in self.indices:
            key = self._make_key(index)
            array[index] = getattr(self, key).instance_for_arguments(
                arguments, ignore_assertions
            )
        return array
