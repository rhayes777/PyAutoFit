import csv
import json
from copy import copy
from typing import List, Optional, Tuple

import numpy as np

from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.abstract import AbstractPriorModel, Path
from autofit.non_linear.samples.sample import Sample


class OptimizerSamples:
    def __init__(
            self,
            model: AbstractPriorModel,
            sample_list: List[Sample],
            time: Optional[float] = None,
    ):
        """
        The `Samples` of a non-linear search, specifically the samples of an search which only provides
        information on the global maximum likelihood solutions, but does not map-out the posterior and thus does
        not provide information on parameter errors.

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """
        self.model = model
        self.sample_list = sample_list
        self.time = time

        self._paths = None
        self._names = None

    def values_for_path(
            self,
            path: Tuple[str]
    ) -> List[float]:
        """
        Returns the value for a variable with a given path
        for each sample in the model
        """
        return [
            sample.kwargs[
                path
            ]
            for sample
            in self.sample_list
        ]

    @property
    def log_evidence(self) -> float:
        return None

    @property
    def paths(self) -> List[Tuple[Path]]:
        """
        A list of paths to unique priors in the same order as prior
        ids (and therefore sample columns)

        Uses hasattr to make backwards compatible
        """
        if not hasattr(self, "_paths") or self._paths is None:
            self._paths = self.model.all_paths
        return self._paths

    @property
    def names(self) -> List[Tuple[str]]:
        """
        A list of names of unique priors in the same order as prior
        ids (and therefore sample columns)

        Uses hasattr to make backwards compatible
        """
        if not hasattr(self, "_names") or self._names is None:
            self._names = self.model.all_names
        return self._names

    @property
    def parameter_lists(self):
        result = list()
        for sample in self.sample_list:
            tuples = self.paths if sample.is_path_kwargs else self.names
            result.append(
                sample.parameter_lists_for_paths(
                    tuples
                )
            )

        return result

    @property
    def total_samples(self):
        return len(self.sample_list)

    @property
    def weight_list(self):
        return [
            sample.weight
            for sample
            in self.sample_list
        ]

    @property
    def log_likelihood_list(self):
        return [
            sample.log_likelihood
            for sample
            in self.sample_list
        ]

    @property
    def log_posterior_list(self):
        return [
            sample.log_posterior
            for sample
            in self.sample_list
        ]

    @property
    def log_prior_list(self):
        return [
            sample.log_prior
            for sample
            in self.sample_list
        ]

    @property
    def parameters_extract(self):
        return np.asarray(self.parameter_lists).T

    @property
    def _headers(self) -> List[str]:
        """
        Headers for the samples table
        """

        return self.model.model_component_and_parameter_names + [
            "log_likelihood",
            "log_prior",
            "log_posterior",
            "weight",
        ]

    @property
    def _rows(self) -> List[List[float]]:
        """
        Rows in the samples table
        """

        log_likelihood_list = self.log_likelihood_list
        log_prior_list = self.log_prior_list
        log_posterior_list = self.log_posterior_list
        weight_list = self.weight_list

        for index, row in enumerate(self.parameter_lists):
            yield row + [
                log_likelihood_list[index],
                log_prior_list[index],
                log_posterior_list[index],
                weight_list[index],
            ]

    def write_table(self, filename: str):
        """
        Write a table of parameters, posteriors, priors and likelihoods.

        Parameters
        ----------
        filename
            Where the table is to be written
        """

        with open(filename, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._headers)
            for row in self._rows:
                writer.writerow(row)

    @property
    def info_json(self):
        return {}

    def info_to_json(self, filename):
        with open(filename, 'w') as outfile:
            json.dump(self.info_json, outfile)

    @property
    def max_log_likelihood_sample(self) -> Sample:
        """
        The index of the sample with the highest log likelihood.
        """
        most_likely_sample = None
        for sample in self.sample_list:
            if most_likely_sample is None or sample.log_likelihood > most_likely_sample.log_likelihood:
                most_likely_sample = sample
        return most_likely_sample

    @property
    def max_log_posterior_sample(self) -> Sample:
        return self.sample_list[
            self.max_log_posterior_index
        ]

    @property
    def max_log_likelihood_vector(self) -> [float]:
        """
        The parameters of the maximum log likelihood sample of the `NonLinearSearch` returned as a list of values.
        """
        sample = self.max_log_likelihood_sample
        return sample.parameter_lists_for_paths(
            self.paths if sample.is_path_kwargs else self.names
        )

    @property
    def max_log_likelihood_instance(self) -> ModelInstance:
        """
        The parameters of the maximum log likelihood sample of the `NonLinearSearch` returned as a model instance.
        """
        return self.max_log_likelihood_sample.instance_for_model(
            self.model
        )

    @property
    def max_log_posterior_index(self) -> int:
        """
        The index of the sample with the highest log posterior.
        """
        return int(np.argmax(self.log_posterior_list))

    @property
    def max_log_posterior_vector(self) -> [float]:
        """
        The parameters of the maximum log posterior sample of the `NonLinearSearch` returned as a list of values.
        """
        return self.parameter_lists[self.max_log_posterior_index]

    @property
    def max_log_posterior_instance(self) -> ModelInstance:
        """
        The parameters of the maximum log posterior sample of the `NonLinearSearch` returned as a model instance.
        """
        return self.model.instance_from_vector(vector=self.max_log_posterior_vector)

    def gaussian_priors_at_sigma(self, sigma: float) -> [list]:
        """
        `GaussianPrior`s of every parameter used to link its inferred values and errors to priors used to sample the
        same (or similar) parameters in a subsequent search, where:

        - The mean is given by maximum log likelihood model values.
        - Their errors are omitted, as this information is not available from an search. When these priors are
        used to link to another search, it will thus automatically use the prior config values.

        Parameters
        -----------
        sigma
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        return list(map(lambda vector: (vector, 0.0), self.max_log_likelihood_vector))

    def instance_from_sample_index(self, sample_index) -> ModelInstance:
        """
        The parameters of an individual sample of the non-linear search, returned as a model instance.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return self.model.instance_from_vector(vector=self.parameter_lists[sample_index])

    def minimise(self) -> "OptimizerSamples":
        """
        A copy of this object with only important samples retained
        """
        samples = copy(self)
        samples.model = None
        samples.sample_list = list({
            self.max_log_likelihood_sample,
            self.max_log_posterior_sample
        })
        return samples
