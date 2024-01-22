import numpy as np
import scipy
from typing import List, Dict

from autofit.non_linear.samples.pdf import SamplesPDF
from .abstract import AbstractInterpolator
from .query import Equality, InterpolatorPath
from autofit.non_linear.analysis.analysis import Analysis
from autofit.non_linear.search.nest.dynesty.search.static import DynestyStatic
from autofit.mapper.prior_model.prior_model import Model
from autofit.mapper.prior_model.collection import Collection
from autofit.mapper.prior.gaussian import GaussianPrior


class LinearRelationship:
    def __init__(self, m: float, c: float):
        """
        Describes a linear relationship between x and y, y = mx + c

        Parameters
        ----------
        m
            The gradient of the relationship
        c
            The y-intercept of the relationship
        """
        self.m = m
        self.c = c

    def __call__(self, x: float) -> float:
        """
        Calculate the value of y for a given value of x
        """
        return self.m * x + self.c

    def __str__(self):
        return f"y = {self.m}x + {self.c}"

    def __repr__(self):
        return str(self)


class CovarianceAnalysis(Analysis):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        inverse_covariance_matrix: np.ndarray,
    ):
        """
        An analysis class that describes a linear relationship between x and y, y = mx + c

        Parameters
        ----------
        x
            The x values (e.g. time)
        y
            The y values. This is a matrix comprising all the variables in the model at each x value
        inverse_covariance_matrix
        """
        self.x = x
        self.y = y
        self.inverse_covariance_matrix = inverse_covariance_matrix

    def _y(self, instance) -> np.ndarray:
        """
        Calculate the y values for a given instance of the model

        Parameters
        ----------
        instance
            An instance of the model

        Returns
        -------
        The y values
        """
        return np.array([relationship(x) for x in self.x for relationship in instance])

    def log_likelihood_function(self, instance) -> float:
        """
        Calculate the log likelihood of the model given an instance of the model

        Parameters
        ----------
        instance
            An instance of the model

        Returns
        -------
        The log likelihood
        """
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
        """
        An interpolator that uses the covariance matrix of a set of samples to find linear
        relationships between variables and some variable on which they depend (e.g. time)

        Parameters
        ----------
        samples_list
            A list of samples from which to calculate the covariance matrix
        """
        self.samples_list = samples_list
        # noinspection PyTypeChecker
        super().__init__([samples.max_log_likelihood() for samples in samples_list])

    def covariance_matrix(self) -> np.ndarray:
        """
        Calculate the covariance matrix of the samples

        This comprises covariance matrices for each sample, subsumed along the diagonal
        """
        matrices = [samples.covariance_matrix for samples in self.samples_list]
        prior_count = self.samples_list[0].model.prior_count
        size = prior_count * len(self.samples_list)
        array = np.zeros((size, size))
        for i, matrix in enumerate(matrices):
            array[
                i * prior_count : (i + 1) * prior_count,
                i * prior_count : (i + 1) * prior_count,
            ] = matrix
        return array

    def inverse_covariance_matrix(self) -> np.ndarray:
        """
        Calculate the inverse covariance matrix of the samples
        """
        covariance_matrix = self.covariance_matrix()
        return scipy.linalg.inv(
            covariance_matrix + 1e-6 * np.eye(covariance_matrix.shape[0])
        )

    @staticmethod
    def _interpolate(x, y, value):
        raise NotImplementedError()

    def _analysis_for_value(self, value: Equality) -> CovarianceAnalysis:
        """
        Create a covariance analysis for a given value. That is an analysis that will
        optimise relationships between each variable and that value.

        Parameters
        ----------
        value
            The value to which the variables are related (e.g. time)

        Returns
        -------
        A linear analysis
        """
        x = []
        y = []
        for sample in sorted(
            self.samples_list,
            key=lambda s: value.path.get_value(s.max_log_likelihood()),
        ):
            # noinspection PyTypeChecker
            x.append(value.path.get_value(sample.max_log_likelihood()))
            y.extend([value for value in sample.max_log_likelihood(as_instance=False)])

        return CovarianceAnalysis(
            np.array(x),
            np.array(y),
            inverse_covariance_matrix=self.inverse_covariance_matrix(),
        )

    def _relationships_for_value(
        self,
        value: Equality,
        path_relationship_map=None,
    ) -> List[LinearRelationship]:
        """
        Calculate the linear relationships between each variable and a given value

        Parameters
        ----------
        value
            The value to which the variables are related (e.g. time)

        Returns
        -------
        A list of linear relationships
        """
        analysis = self._analysis_for_value(value)
        model = self.model(path_relationship_map=path_relationship_map or {})
        optimizer = DynestyStatic()
        result = optimizer.fit(model=model, analysis=analysis)
        return result.instance

    def __getitem__(self, value: Equality) -> float:
        return self.get(value)

    def get(
        self,
        value: Equality,
        path_relationship_map: Dict[InterpolatorPath, Model] = None,
    ) -> float:
        """
        Calculate the value of the variable for a given value of the variable to which it is related

        Parameters
        ----------
        path_relationship_map
            Specifies which model should be used for each relationship. Each parameter in the model
            is described by a LinearRelationship with GaussianPriors by default.
        value
            The value to which the variables are related (e.g. time)

        Returns
        -------
        The value of the variable at the given time
        """
        relationships = self._relationships_for_value(
            value,
            path_relationship_map=path_relationship_map,
        )
        model = self._single_model
        arguments = {
            prior: relationship(value.value)
            for prior, relationship in zip(
                model.priors_ordered_by_id,
                relationships,
            )
        }
        return model.instance_for_arguments(arguments)

    def _max_likelihood_samples_list(self) -> SamplesPDF:
        """
        Find the samples with the highest log likelihood
        """
        return max(
            self.samples_list, key=lambda s: s.max_log_likelihood_sample.log_likelihood
        )

    @property
    def _single_model(self):
        """
        The model from the point in the time series that gave the highest likelihood
        """
        return self._max_likelihood_samples_list().model

    def model(self, path_relationship_map=None) -> Collection:
        """
        Create a model that describes the linear relationships between each variable and the variable to which it is
        related
        """
        models = []
        single_model = self._single_model
        path_relationship_map = path_relationship_map or {}
        for prior in single_model.priors_ordered_by_id:
            path = single_model.path_for_prior(prior)
            if path in path_relationship_map:
                models.append(path_relationship_map[path])
                continue

            mean = prior.mean
            models.append(
                Model(
                    LinearRelationship,
                    m=GaussianPrior(mean=mean, sigma=mean),
                    c=GaussianPrior(mean=mean, sigma=mean),
                )
            )
        return Collection(models)
