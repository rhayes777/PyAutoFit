import numpy as np

from autofit.graphical.messages.abstract import AbstractMessage


class FixedMessage(AbstractMessage):
    log_base_measure = 0

    def __init__(self, value, log_norm=0.):
        self._value = value
        super().__init__(
            (value,),
            log_norm=log_norm
        )

    @property
    def natural_parameters(self):
        return self.parameters

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        return natural_parameters,

    @staticmethod
    def to_canonical_form(x):
        return x

    @property
    def log_partition(self):
        return 0.

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats):
        return suff_stats

    def sample(self, n_samples):
        """
        Rely on array broadcasting to get fixed values to
        calculate correctly
        """
        return np.array(self.parameters)

    def logpdf(self, x):
        return np.zeros_like(x)

    @property
    def mean(self):
        return self._value

    @property
    def variance(self):
        return np.zeros_like(self.mean)

    def _no_op(self, *other, **kwargs):
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
