from typing import Tuple

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
