from typing import Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import Sample
from autofit.non_linear.samples.nest import NestSamples


class UltraNestSamples(NestSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            results,
            number_live_points: int,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):
        """
        The *Output* classes in **PyAutoFit** provide an interface between the results of a `NonLinearSearch` (e.g.
        as files on your hard-disk) and Python.

        For example, the output class can be used to load an instance of the best-fit model, get an instance of any
        individual sample by the `NonLinearSearch` and return information on the likelihoods, errors, etc.

        The Bayesian log evidence estimated by the nested sampling algorithm.

        Parameters
        ----------
        model : af.ModelMapper
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        number_live_points : int
            The number of live points used by the nested sampler.
        log_evidence
            The log of the Bayesian evidence estimated by the nested sampling algorithm.
        """

        self.results = results
        self._number_live_points = number_live_points

        parameters = self.results["weighted_samples"]["points"]
        log_likelihood_list = self.results["weighted_samples"]["logl"]
        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameters
        ]
        weight_list = self.results["weighted_samples"]["weights"]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        super().__init__(
            model=model,
            sample_list=sample_list,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

    @property
    def number_live_points(self):
        return self._number_live_points

    @property
    def total_samples(self):
        return self.results["ncall"]

    @property
    def log_evidence(self):
        return self.results["logz"]
