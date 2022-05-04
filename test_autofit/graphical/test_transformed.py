import functools
from typing import Tuple, Optional

import numpy as np

from autofit.messages import AbstractMessage, NormalMessage, UniformNormalMessage
from autofit.messages.transform import phi_transform


def transform_argument(func):
    @functools.wraps(func)
    def wrapper(self, argument):
        return func(
            self,
            self.transform_value(argument)
        )

    return wrapper


class TransformedMessage(AbstractMessage):
    def __init__(self, message, transforms):
        super().__init__()
        self.message = message
        self.transforms = transforms

    def transform_value(self, value):
        for transform in self.transforms:
            value = transform.transform(value)
        return value

    @transform_argument
    def calc_log_base_measure(self, x) -> np.ndarray:
        return self.message.calc_log_base_measure(x)

    @transform_argument
    def factor(self, x):
        return self.message.factor(x)

    def natural_parameters(self):
        pass

    def sample(self, n_samples: Optional[int] = None):
        pass

    def invert_natural_parameters(self, natural_parameters: np.ndarray) -> Tuple[np.ndarray, ...]:
        pass

    def to_canonical_form(self, x: np.ndarray) -> np.ndarray:
        pass

    def log_partition(self) -> np.ndarray:
        pass

    def variance(self) -> np.ndarray:
        pass

    def invert_sufficient_statistics(self, sufficient_statistics: np.ndarray) -> np.ndarray:
        pass


def _test_transform():
    transformed_message = TransformedMessage(
        NormalMessage(
            mean=0.0,
            sigma=1.0
        ),
        [
            phi_transform
        ]
    )
    uniform_normal_message = UniformNormalMessage(
        mean=0.0,
        sigma=1.0,
    )
    assert uniform_normal_message.factor(0.5) == transformed_message.factor(0.5)
