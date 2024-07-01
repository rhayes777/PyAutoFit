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

        for key in np.ndindex(*shape):
            suffix = "_".join(map(str, key))
            setattr(self, f"prior_{suffix}", prior.new())

    def _instance_for_arguments(
        self,
        arguments: Dict[Prior, float],
        ignore_assertions: bool = False,
    ):
        return np.array(
            [
                [
                    arguments[getattr(self, f"prior_{i}_{j}")]
                    for j in range(self.shape[1])
                ]
                for i in range(self.shape[0])
            ]
        )
