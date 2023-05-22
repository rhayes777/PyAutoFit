import numpy as np
import scipy
from typing import List

from scipy.optimize import curve_fit
from autofit.non_linear.samples.pdf import SamplesPDF
from .abstract import AbstractInterpolator
from .query import Equality


class CovarianceInterpolator(AbstractInterpolator):
    def __init__(
        self,
        samples_list: List[SamplesPDF],
    ):
        self.samples_list = samples_list
        # noinspection PyTypeChecker
        super().__init__([samples.max_log_likelihood() for samples in samples_list])

    @property
    def covariance_matrix(self):
        matrices = [samples.covariance_matrix() for samples in self.samples_list]
        prior_count = self.samples_list[0].model.prior_count
        size = prior_count * len(self.samples_list)
        array = np.zeros((size, size))
        for i, matrix in enumerate(matrices):
            array[
                i * prior_count : (i + 1) * prior_count,
                i * prior_count : (i + 1) * prior_count,
            ] = matrix
        return array

    @property
    def inverse_covariance_matrix(self):
        return scipy.linalg.inv(self.covariance_matrix)

    @staticmethod
    def _interpolate(x, y, value):
        pass

    def __getitem__(self, value: Equality):
        x = [value.path.get_value(instance) for instance in self.instances]

        def func(x, *args):
            return x

        curve = curve_fit(
            func, x, self._y, p0=2 * len(x) * [1], sigma=self.covariance_matrix
        )
        print(curve)

    @property
    def _y(self):
        return np.array(
            [
                value
                for samples in self.samples_list
                for value in samples.max_log_likelihood(as_instance=False)
            ]
        )
