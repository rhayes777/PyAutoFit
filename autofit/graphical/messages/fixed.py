from typing import Optional, Tuple
import numpy as np

from autofit.graphical.messages.abstract import AbstractMessage


class FixedMessage(AbstractMessage):
    log_base_measure = 0

    def __init__(self, value: np.ndarray, log_norm: np.ndarray=0.):
        self.value = value
        super().__init__(
            (value,),
            log_norm=log_norm
        )

    @property
    def natural_parameters(self) -> Tuple[np.ndarray, ...]:
        return self.parameters

    @staticmethod
    def invert_natural_parameters(natural_parameters: np.ndarray
    ) -> Tuple[np.ndarray]:
        return natural_parameters,

    @staticmethod
    def to_canonical_form(x: np.ndarray) -> np.ndarray:
        return x

    @property
    def log_partition(self) -> np.ndarray:
        return 0.

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats: np.ndarray
    ) -> np.ndarray:
        return suff_stats

    def sample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Rely on array broadcasting to get fixed values to
        calculate correctly
        """
        if n_samples is None:
            return self.value
        else:
            return np.array([self.value])

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    @property
    def mean(self) -> np.ndarray:
        return self.value

    @property
    def variance(self) -> np.ndarray:
        return np.zeros_like(self.mean)

    def _no_op(self, *other, **kwargs) -> 'FixedMessage':
        """
        'no-op' operation

        In many operations fixed messages should just
        return themselves
        """
        return self

    project = _no_op
    from_mode = _no_op
    __pow__ = _no_op
    __mul__ = _no_op
    __div__ = _no_op
    default = _no_op
    _multiply = _no_op
    _divide = _no_op
    sum_natural_parameters = _no_op
    sub_natural_parameters = _no_op

    def kl(self, dist: "FixedMessage") -> float:
        return 0.
