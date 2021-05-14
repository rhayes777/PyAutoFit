from typing import Optional

import numpy as np
from dynesty.results import Results

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import NestSamples, Sample


class DynestySamples(NestSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            results: Results,
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
        log_evidence : float
            The log of the Bayesian evidence estimated by the nested sampling algorithm.
        """

        super().__init__(
            model=model,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

        self.results = results
        self._samples = None
        self._number_live_points = number_live_points

    @property
    def samples(self):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Emcee, all quantities are extracted via the hdf5 backend of results.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the `NonLinearSearch` chains,
            etc.
        """

        if self._samples is not None:
            return self._samples

        parameter_lists = self.results.samples.tolist()
        log_prior_list = [
            sum(self.model.log_prior_list_from_vector(vector=vector)) for vector in parameter_lists
        ]
        log_likelihood_list = list(self.results.logl)

        try:
            weight_list = list(
                np.exp(np.asarray(self.results.logwt) - self.results.logz[-1])
            )
        except:
            weight_list = self.results["weights"]

        self._samples = Sample.from_lists(
            model=self.model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return self._samples

    @property
    def number_live_points(self):
        return self._number_live_points

    @property
    def total_samples(self):
        return int(np.sum(self.results.ncall))

    @property
    def log_evidence(self):
        return np.max(self.results.logz)
