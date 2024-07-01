from typing import Tuple, Dict, Optional

from autoconf.dictable import from_dict
from .abstract import AbstractPriorModel
from autofit.mapper.prior.abstract import Prior
import numpy as np


class Array(AbstractPriorModel):
    def __init__(
        self,
        shape: Tuple[int, ...],
        prior: Optional[Prior] = None,
    ):
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
        self.indices = list(np.ndindex(*shape))

        if prior is not None:
            for index in self.indices:
                self[index] = prior.new()

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
            value = self[index]
            try:
                value = value.instance_for_arguments(
                    arguments,
                    ignore_assertions,
                )
            except AttributeError:
                pass

            array[index] = value
        return array

    def __setitem__(self, key, value):
        setattr(self, self._make_key(key), value)

    def __getitem__(self, key):
        return getattr(self, self._make_key(key))

    @classmethod
    def from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ):
        arguments = d["arguments"]
        shape = from_dict(arguments["shape"])
        array = cls(shape)
        for key, value in arguments.items():
            if key.startswith("prior"):
                setattr(array, key, from_dict(value))

        return array
