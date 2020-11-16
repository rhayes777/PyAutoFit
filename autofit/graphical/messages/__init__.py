from typing import (
    Dict, Tuple, Iterator
)

import numpy as np

from .abstract import AbstractMessage
from .fixed import FixedMessage
from .gamma import GammaMessage
from .normal import NormalMessage


def map_dists(dists: Dict[str, AbstractMessage],
              values: Dict[str, np.ndarray],
              _call: str = 'logpdf'
              ) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Calls a method (default: logpdf) for each Message in dists
    on the corresponding value in values
    """
    for v in dists.keys() & values.keys():
        dist = dists[v]
        if isinstance(dist, AbstractMessage):
            yield v, getattr(dist, _call)(values[v])
