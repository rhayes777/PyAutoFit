from typing import Tuple, Dict, Optional, Union

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
    def _make_key(index: Tuple[int, ...]) -> str:
        """
        Make a key for the prior.

        This is so an index (e.g. (1, 2)) can be used to access a
        prior (e.g. prior_1_2).

        Parameters
        ----------
        index
            The index of an element in an array.

        Returns
        -------
        The attribute name for the prior.
        """
        if isinstance(index, int):
            suffix = str(index)
        else:
            suffix = "_".join(map(str, index))
        return f"prior_{suffix}"

    def _instance_for_arguments(
        self,
        arguments: Dict[Prior, float],
        ignore_assertions: bool = False,
    ) -> np.ndarray:
        """
        Create an array where the prior at each index is replaced with the
        a concrete value.

        Parameters
        ----------
        arguments
            The arguments to replace the priors with.
        ignore_assertions
            Whether to ignore assertions in the priors.

        Returns
        -------
        The array with the priors replaced.
        """
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

    def __setitem__(
        self,
        index: Union[int, Tuple[int, ...]],
        value: Union[float, Prior],
    ):
        """
        Set the value at an index.

        Parameters
        ----------
        index
            The index of the prior.
        value
            The new value.
        """
        setattr(
            self,
            self._make_key(index),
            value,
        )

    def __getitem__(
        self,
        index: Union[int, Tuple[int, ...]],
    ) -> Union[float, Prior]:
        """
        Get the value at an index.

        Parameters
        ----------
        index
            The index of the value.

        Returns
        -------
        The value at the index.
        """
        return getattr(
            self,
            self._make_key(index),
        )

    @classmethod
    def from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ) -> "Array":
        """
        Create an array from a dictionary.

        Parameters
        ----------
        d
            The dictionary.
        reference
            A dictionary of references.
        loaded_ids
            A dictionary of loaded ids.

        Returns
        -------
        The array.
        """
        arguments = d["arguments"]
        shape = from_dict(arguments["shape"])
        array = cls(shape)
        for key, value in arguments.items():
            if key.startswith("prior"):
                setattr(array, key, from_dict(value))

        return array
