from collections import defaultdict

import numpy as np
import scipy
from typing import List

from autofit.non_linear.samples.pdf import SamplesPDF
from .abstract import AbstractInterpolator
from .query import Equality
from autofit.non_linear.analysis.analysis import Analysis
from autofit.non_linear.nest.dynesty.static import DynestyStatic
from autofit.mapper.prior_model.prior_model import Model
from autofit.mapper.prior_model.collection import Collection
from autofit.mapper.prior.gaussian import GaussianPrior


class LinearRelationship:
    def __init__(self, m: float, c: float):
        self.m = m
        self.c = c

    def __call__(self, x):
        return self.m * x + self.c


class LinearAnalysis(Analysis):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        inverse_covariance_matrix: np.ndarray,
    ):
        self.x = x
        self.y = y
        self.inverse_covariance_matrix = inverse_covariance_matrix

    def _y(self, instance):
        return np.array([relationship(x) for x in self.x for relationship in instance])

    def log_likelihood_function(self, instance):
        return -0.5 * (
            np.dot(
                self.y - self._y(instance),
                np.dot(self.inverse_covariance_matrix, self.y - self._y(instance)),
            )
        )


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

    def _linear_analysis_for_value(self, value: Equality):
        x = []
        y = []
        for sample in sorted(
            self.samples_list,
            key=lambda s: value.path.get_value(s.max_log_likelihood()),
        ):
            # noinspection PyTypeChecker
            x.append(value.path.get_value(sample.max_log_likelihood()))
            y.extend([value for value in sample.max_log_likelihood(as_instance=False)])

        return LinearAnalysis(
            np.array(x),
            np.array(y),
            inverse_covariance_matrix=self.inverse_covariance_matrix,
        )

    def _relationships_for_value(self, value: Equality):
        analysis = self._linear_analysis_for_value(value)
        model = self.model
        optimizer = DynestyStatic()
        result = optimizer.fit(model=model, analysis=analysis)
        return result.instance

    def __getitem__(self, value: Equality):
        relationships = self._relationships_for_value(value)
        model = self._single_model
        arguments = {
            prior: relationship(value.value)
            for prior, relationship in zip(
                model.priors_ordered_by_id,
                relationships,
            )
        }
        return model.instance_for_arguments(arguments)

    def _max_likelihood_samples_list(self):
        return max(self.samples_list, key=lambda s: max(s.log_likelihood_list))

    @property
    def _single_model(self):
        return self._max_likelihood_samples_list().model

    @property
    def model(self):
        models = []
        for prior in self._single_model.priors_ordered_by_id:
            mean = prior.mean
            models.append(
                Model(
                    LinearRelationship,
                    m=GaussianPrior(mean=mean, sigma=mean),
                    c=GaussianPrior(mean=mean, sigma=mean),
                )
            )
        return Collection(models)
