import logging
import math
import os

import numpy as np
import pymultinest

from autofit import conf, exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.optimize.non_linear.nested_sampling.nested_sampler import (
    NestedSampler,
    NestedSamplerOutput,
)
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.paths import Paths

logger = logging.getLogger(__name__)


class MultiNest(NestedSampler):
    def __init__(self, paths=None, sigma=3, run=pymultinest.run):
        """
        Class to setup and run a MultiNest non-linear search.

        For a full description of MultiNest and its Python wrapper PyMultiNest, checkout its Github and documentation
        webpages:

        https://github.com/JohannesBuchner/MultiNest
        https://github.com/JohannesBuchner/PyMultiNest
        http://johannesbuchner.github.io/PyMultiNest/index.html#

        Parameters
        ----------
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search chains,
            backups, etc.
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        terminate_at_acceptance_ratio : bool
            If *True*, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *NestedSampler* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            *True* (see *NestedSampler* for a full description of this feature).

        """
        super().__init__(paths=paths, sigma=sigma)

        self.importance_nested_sampling = self.config(
            "importance_nested_sampling", bool
        )
        self.multimodal = self.config("multimodal", bool)
        self.const_efficiency_mode = self.config("const_efficiency_mode", bool)
        self.n_live_points = self.config("n_live_points", int)
        self.evidence_tolerance = self.config("evidence_tolerance", float)
        self.sampling_efficiency = self.config("sampling_efficiency", float)
        self.n_iter_before_update = self.config("n_iter_before_update", int)
        self.null_log_evidence = self.config("null_log_evidence", float)
        self.max_modes = self.config("max_modes", int)
        self.mode_tolerance = self.config("mode_tolerance", float)
        self.seed = self.config("seed", int)
        self.verbose = self.config("verbose", bool)
        self.resume = self.config("resume", bool)
        self.context = self.config("context", int)
        self.write_output = self.config("write_output", bool)
        self.log_zero = self.config("log_zero", float)
        self.max_iter = self.config("max_iter", int)
        self.init_MPI = self.config("init_MPI", bool)
        self.terminate_at_acceptance_ratio = self.config(
            "terminate_at_acceptance_ratio", bool
        )
        self.acceptance_ratio_threshold = self.config(
            "acceptance_ratio_threshold", float
        )

        self.run = run

        logger.debug("Creating MultiNest NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the multinest non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma = self.sigma
        copy.run = self.run
        copy.importance_nested_sampling = self.importance_nested_sampling
        copy.multimodal = self.multimodal
        copy.const_efficiency_mode = self.const_efficiency_mode
        copy.n_live_points = self.n_live_points
        copy.evidence_tolerance = self.evidence_tolerance
        copy.sampling_efficiency = self.sampling_efficiency
        copy.n_iter_before_update = self.n_iter_before_update
        copy.null_log_evidence = self.null_log_evidence
        copy.max_modes = self.max_modes
        copy.mode_tolerance = self.mode_tolerance
        copy.seed = self.seed
        copy.verbose = self.verbose
        copy.resume = self.resume
        copy.context = self.context
        copy.write_output = self.write_output
        copy.log_zero = self.log_zero
        copy.max_iter = self.max_iter
        copy.init_MPI = self.init_MPI
        copy.terminate_at_acceptance_ratio = self.terminate_at_acceptance_ratio
        copy.acceptance_ratio_threshold = self.acceptance_ratio_threshold
        return copy

    def _simple_fit(self, model: AbstractPriorModel, fitness_function) -> Result:
        """
        Fit a model using MultiNest and a function that returns a likelihood from instances of that model.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        fitness_function
            A function that fits this model to the data, returning the likelihood of the fit.

        Returns
        -------
        A result object comprising the best-fit model instance, likelihood and an *Output* class that enables analysis
        of the full chains used by the fit.
        """
        multinest_output = MultiNestOutput(model, self.paths)

        def prior(cube, ndim, nparams):
            # NEVER EVER REFACTOR THIS LINE! Haha.

            phys_cube = model.vector_from_unit_vector(unit_vector=cube)

            for i in range(len(phys_cube)):
                cube[i] = phys_cube[i]

            return cube

        stagger_resampling_likelihood = conf.instance.non_linear.get(
            "MultiNest", "stagger_resampling_likelihood", bool
        )
        stagger_resampling_value = conf.instance.non_linear.get(
            "MultiNest", "stagger_resampling_value", float
        )

        class Fitness:
            def __init__(self):
                """
                Fitness function that only handles resampling
                """
                self.stagger_accepted_samples = 0
                self.resampling_likelihood = conf.instance.non_linear.get(
                    "MultiNest", "null_log_evidence", float
                )

            def __call__(self, cube, ndim, nparams, lnew):
                """
                This call converts a vector of physical values then determines a fit.

                If an exception is thrown it handles resampling.
                """
                try:
                    return fitness_function(model.instance_from_vector(cube))
                except exc.FitException:
                    if not stagger_resampling_likelihood:
                        likelihood = -np.inf
                    else:
                        if self.stagger_accepted_samples < 10:
                            self.stagger_accepted_samples += 1
                            self.resampling_likelihood += stagger_resampling_value
                            likelihood = self.resampling_likelihood
                        else:
                            likelihood = (
                                -1.0 * np.abs(self.resampling_likelihood) * 10.0
                            )
                    return likelihood

        self.run(
            Fitness().__call__,
            prior,
            model.prior_count,
            outputfiles_basename="{}/multinest".format(self.paths.path),
            n_live_points=self.n_live_points,
            const_efficiency_mode=self.const_efficiency_mode,
            importance_nested_sampling=self.importance_nested_sampling,
            evidence_tolerance=self.evidence_tolerance,
            sampling_efficiency=self.sampling_efficiency,
            null_log_evidence=self.null_log_evidence,
            n_iter_before_update=self.n_iter_before_update,
            multimodal=self.multimodal,
            max_modes=self.max_modes,
            mode_tolerance=self.mode_tolerance,
            seed=self.seed,
            verbose=self.verbose,
            resume=self.resume,
            context=self.context,
            write_output=self.write_output,
            log_zero=self.log_zero,
            max_iter=self.max_iter,
            init_MPI=self.init_MPI,
        )
        self.paths.backup()

        instance = multinest_output.most_likely_instance
        multinest_output.output_results(during_analysis=False)
        return Result(
            instance=instance,
            likelihood=multinest_output.maximum_log_likelihood,
            output=multinest_output,
            previous_model=model,
            gaussian_tuples=multinest_output.gaussian_priors_at_sigma(self.sigma),
        )

    def output_from_model(self, model, paths):
        """Create this non-linear search's output class from the model and paths.

        This function is required by the aggregator, so it knows which output class to generate an instance of."""
        return MultiNestOutput(model=model, paths=paths)


class MultiNestOutput(NestedSamplerOutput):
    @property
    def pdf(self):
        """An interface to *GetDist* which can be used for analysing and visualizing the non-linear search chains.

        *GetDist* can only be used when chains are converged enough to provide a smooth PDF and this convergence is
        checked using the *pdf_converged* bool before *GetDist* is called.

        https://github.com/cmbant/getdist
        https://getdist.readthedocs.io/en/latest/

        For *MultiNest*, chains are passed to *GetDist* via the multinest.txt file, which contains the physical model
        parameters of every accepted sample and its sampling probabilitiy which is used as the weight.
        """

        import getdist

        try:
            return getdist.mcsamples.loadMCSamples(
                self.paths.backup_path + "/multinest"
            )
        except IOError or OSError or ValueError or IndexError:
            raise Exception

    @property
    def pdf_converged(self) -> bool:
        """ To analyse and visualize chains using *GetDist*, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's chains are sufficiently
        converged for *GetDist* use.

        For *MultiNest*, during initial sampling one accepted live point typically has > 99% of the probabilty as its
        likelihood is significantly higher than all other points. Convergence is only achieved late in sampling when
        all live points have similar likelihood and sampling probabilities."""
        try:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            if len(densities_1d) == 0:
                return False

            return True
        except Exception:
            return False

    @property
    def number_live_points(self) -> int:
        """The number of live points used by the nested sampler."""
        return len(self.phys_live_points)

    @property
    def total_samples(self) -> int:
        """The total number of samples performed by the non-linear search.

        For MulitNest, this includes all accepted and rejected samples, and is loaded from the "multinestresume" file.
        """
        resume = open(self.paths.file_resume)

        resume.seek(1)
        resume.read(19)
        return int(resume.read(8))

    @property
    def total_accepted_samples(self) -> int:
        """The total number of accepted samples performed by the non-linear search.

        For MulitNest, this is loaded from the "multinestresume" file.
        """
        resume = open(self.paths.file_resume)

        resume.seek(1)
        resume.read(8)
        return int(resume.read(10))

    @property
    def acceptance_ratio(self) -> float:
        """The ratio of accepted samples to total samples."""
        return self.total_accepted_samples / self.total_samples

    @property
    def maximum_log_likelihood(self) -> float:
        """The maximum log likelihood value of the non-linear search, corresponding to the best-fit model.

        For MultiNest, this is read from the "multinestsummary.txt" file. """
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=2, offset=112
            )[1]
        except FileNotFoundError:
            return max([point[-1] for point in self.phys_live_points])

    @property
    def evidence(self) -> float:
        """The Bayesian evidence estimated by the nested sampling algorithm.

        For MultiNest, this is read from the "multinestsummary.txt" file."""
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=2, offset=112
            )[0]
        except FileNotFoundError:
            return None

    @property
    def most_likely_index(self) -> int:
        """The index of the accepted sample with the highest likelihood, e.g. that of best-fit / most_likely model."""
        return int(np.argmax([point[-1] for point in self.phys_live_points]))

    @property
    def most_likely_vector(self) -> [float]:
        """ The best-fit model sampled by the non-linear search (corresponding to the maximum log-likelihood), returned
        as a list of values.

        The vector is read from the MulitNest file "multinestsummary.txt, which stores the parameters of the most
        likely model in the second half of entries."""
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=self.model.prior_count, offset=56
            )
        except FileNotFoundError:
            return self.phys_live_points[self.most_likely_index][0:-1]

    @property
    def most_probable_vector(self) -> [float]:
        """ The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values.

        The vector is read from the MulitNest file "multinestsummary.txt, which stores the parameters of the most
        probable model in the first half of entries.
        """
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=self.model.prior_count, offset=0
            )
        except FileNotFoundError:
            return self.most_likely_vector

    def vector_at_sigma(self, sigma) -> [float]:
        """ The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For *MultiNest*, this is estimated using *GetDist* if the chains have converged, by sampling the density
        function at an input PDF %. If not converged, a crude estimate using the range of values of the current
        physical live points is used.

        Parameters
        ----------
        sigma : float
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF)."""
        limit = math.erf(0.5 * sigma * math.sqrt(2))

        if self.pdf_converged:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            return list(map(lambda p: p.getLimits(limit), densities_1d))
        else:

            parameters_min = [
                min(self.phys_live_points_of_param(param_index=param_index))
                for param_index in range(self.model.prior_count)
            ]
            parameters_max = [
                max(self.phys_live_points_of_param(param_index=param_index))
                for param_index in range(self.model.prior_count)
            ]

            return [
                (parameters_min[index], parameters_max[index])
                for index in range(len(parameters_min))
            ]

    def vector_from_sample_index(self, sample_index) -> [float]:
        """The model parameters of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return list(self.pdf.samples[sample_index])

    def weight_from_sample_index(self, sample_index) -> float:
        """The weight of an individual sample of the non-linear search.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return self.pdf.weights[sample_index]

    def likelihood_from_sample_index(self, sample_index) -> float:
        """The likelihood of an individual sample of the non-linear search.

        NOTE: GetDist reads the log likelihood from the weighted_sample.txt file (column 2), which are defined as \
        -2.0*likelihood. This routine converts these back to likelihood.

        Parameters
        ----------
        sample_index : int
            The index of the sample in the non-linear search, e.g. 0 gives the first sample.
        """
        return -0.5 * self.pdf.loglikes[sample_index]

    @property
    def phys_live_points(self) -> [[float]]:
        """Open the MultiNest "multinestphys_live_points" file, read it and return all physical live point models as
        a list of 1D lists."""
        phys_live = open(self.paths.file_phys_live)

        live_points = 0
        for line in phys_live:
            live_points += 1

        phys_live.seek(0)

        phys_live_points = []

        for line in range(live_points):
            vector = []
            for param in range(self.model.prior_count + 1):
                vector.append(float(phys_live.read(28)))
            phys_live.readline()
            phys_live_points.append(vector)

        phys_live.close()

        return phys_live_points

    def phys_live_points_of_param(self, param_index) -> [float]:
        """Return all parameter values of a given parameter for all of the current physical live points.

        These are read from the "multinestphys_live_points" file."""
        return [point[param_index] for point in self.phys_live_points]

    def read_list_of_results_from_summary_file(self, number_entries, offset) -> [float]:
        """Read a list of results from the "multinestsummary.txt" file, which stores information on the MulitNest run
        incuding the most likely and most probable models.

        This file stores the results as a text file where different columns correspnd to different models."""

        summary = open(self.paths.file_summary)
        summary.read(2 + offset * self.model.prior_count)
        vector = []
        for param in range(number_entries):
            vector.append(float(summary.read(28)))

        summary.close()

        return vector
