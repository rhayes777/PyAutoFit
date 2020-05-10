import logging

import numpy as np

from autofit import exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.optimize.non_linear import samples
from autofit.optimize.non_linear.nested_sampling import nested_sampler as ns
from autofit.optimize.non_linear import non_linear as nl
from autofit.text import samples_text

logger = logging.getLogger(__name__)


class MultiNest(ns.NestedSampler):
    def __init__(
        self,
        paths=None,
        sigma=3,
        n_live_points=None,
        sampling_efficiency=None,
        const_efficiency_mode=None,
        evidence_tolerance=None,
        multimodal=None,
        importance_nested_sampling=None,
        n_iter_before_update=None,
        null_log_evidence=None,
        max_modes=None,
        mode_tolerance=None,
        seed=None,
        verbose=None,
        resume=None,
        context=None,
        write_output=None,
        log_zero=None,
        max_iter=None,
        init_MPI=None,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
        stagger_resampling_likelihood=None,
    ):
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
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search samples,
            backups, etc.
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        """

        super().__init__(
            paths=paths,
            sigma=sigma,
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
        )

        self.n_live_points = (
            self.config("search", "n_live_points", int)
            if n_live_points is None
            else n_live_points
        )
        self.sampling_efficiency = (
            self.config("search", "sampling_efficiency", float)
            if sampling_efficiency is None
            else sampling_efficiency
        )
        self.const_efficiency_mode = (
            self.config("search", "const_efficiency_mode", bool)
            if const_efficiency_mode is None
            else const_efficiency_mode
        )
        self.evidence_tolerance = (
            self.config("search", "evidence_tolerance", float)
            if evidence_tolerance is None
            else evidence_tolerance
        )
        self.multimodal = (
            multimodal or self.config("search", "multimodal", bool)
            if multimodal is None
            else multimodal
        )
        self.importance_nested_sampling = (
            self.config("search", "importance_nested_sampling", bool)
            if importance_nested_sampling is None
            else importance_nested_sampling
        )
        self.max_modes = (
            self.config("search", "max_modes", int) if max_modes is None else max_modes
        )
        self.mode_tolerance = (
            self.config("search", "mode_tolerance", float)
            if mode_tolerance is None
            else mode_tolerance
        )
        self.max_iter = self.config("search", "max_iter", int) if max_iter is None else max_iter
        self.n_iter_before_update = (
            self.config("settings", "n_iter_before_update", int)
            if n_iter_before_update is None
            else n_iter_before_update
        )
        self.null_log_evidence = (
            self.config("settings", "null_log_evidence", float)
            if null_log_evidence is None
            else null_log_evidence
        )
        self.seed = self.config("settings", "seed", int) if seed is None else seed
        self.verbose = self.config("settings", "verbose", bool) if verbose is None else verbose
        self.resume = self.config("settings", "resume", bool) if resume is None else resume
        self.context = self.config("settings", "context", int) if context is None else context
        self.write_output = (
            self.config("settings", "write_output", bool) if write_output is None else write_output
        )
        self.log_zero = self.config("settings", "log_zero", float) if log_zero is None else log_zero
        self.init_MPI = self.config("settings", "init_MPI", bool) if init_MPI is None else init_MPI

        self.stagger_resampling_likelihood = (
            self.config("settings", "stagger_resampling_likelihood", bool)
            if stagger_resampling_likelihood is None
            else stagger_resampling_likelihood
        )

        logger.debug("Creating MultiNest NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the multinest non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma = self.sigma
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

    class Fitness(ns.NestedSampler.Fitness):
        def __init__(
            self,
            paths,
            analysis,
            model,
            samples_from_model,
            stagger_resampling_likelihood,
            terminate_at_acceptance_ratio,
            acceptance_ratio_threshold,
        ):

            super().__init__(
                paths=paths,
                analysis=analysis,
                model=model,
                samples_from_model=samples_from_model,
                terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
                acceptance_ratio_threshold=acceptance_ratio_threshold,
            )

            self.stagger_resampling_likelihood = stagger_resampling_likelihood
            self.stagger_accepted_samples = 0
            self.resampling_likelihood = -1.0e99

        def __call__(self, params, *kwargs):

            self.check_terminate_sampling()

            try:

                instance = self.model.instance_from_vector(vector=params)
                return self.fit_instance(instance)

            except exc.FitException:

                return self.stagger_resampling_log_likelihood()

        def stagger_resampling_log_likelihood(self):
            """By default, when a fit raises an exception a log likelihood of -np.inf is returned, which leads the
            sampler to discard the sample.

            However, we found that this causes memory issues when running PyMultiNest. Therefore, we 'hack' a solution
            by not returning -np.inf (which leads the sample to be discarded) but instead a large negative float which
            is treated as a real sample (and does not lead too memory issues). The value returned is staggered to avoid
            all initial samples returning the same log likelihood and the non-linear search terminating."""

            if not self.stagger_resampling_likelihood:

                return -np.inf

            else:
                if self.stagger_accepted_samples < 10:

                    self.stagger_accepted_samples += 1
                    self.resampling_likelihood += 1e90

                    return self.resampling_likelihood

                else:

                    return -1.0 * np.abs(self.resampling_likelihood) * 10.0

    def _fit(self, model: AbstractPriorModel, analysis) -> nl.Result:
        """
        Fit a model using MultiNest and a function that returns a log likelihood from instances of that model.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        fitness_function
            A function that fits this model to the data, returning the log likelihood of the fit.

        Returns
        -------
        A result object comprising the best-fit model instance, log_likelihood and an *Output* class that enables analysis
        of the full samples used by the fit.
        """

        def prior(cube, ndim, nparams):
            # NEVER EVER REFACTOR THIS LINE! Haha.

            phys_cube = model.vector_from_unit_vector(unit_vector=cube)

            for i in range(len(phys_cube)):
                cube[i] = phys_cube[i]

            return cube

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis
        )

        import pymultinest

        pymultinest.run(
            fitness_function,
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

        samples = self.samples_from_model(model=model)

        samples_text.results_to_file(
            samples=samples, file_results=self.paths.file_results, during_analysis=False
        )

        return nl.Result(samples=samples, previous_model=model)

    def fitness_function_from_model_and_analysis(self, model, analysis):

        return MultiNest.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from_model,
            stagger_resampling_likelihood=self.stagger_resampling_likelihood,
            terminate_at_acceptance_ratio=self.terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=self.acceptance_ratio_threshold,
        )

    def samples_from_model(self, model: AbstractPriorModel):
        """Create a *Samples* object from this non-linear search's output files on the hard-disk and model.

        For MulitNest, this requires us to load:

            - The parameter samples, log likelihood values and weights from the multinest.txt file.
            - The total number of samples (e.g. accepted + rejected) from resume.dat.
            - The log evidence of the model-fit from the multinestsummary.txt file (if this is not yet estimated a
              value of -1.0e99 is used.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        parameters = parameters_from_file_weighted_samples(
            file_weighted_samples=self.paths.file_weighted_samples,
            prior_count=model.prior_count,
        )

        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]

        log_likelihoods = log_likelihoods_from_file_weighted_samples(
            file_weighted_samples=self.paths.file_weighted_samples
        )

        weights = weights_from_file_weighted_samples(
            file_weighted_samples=self.paths.file_weighted_samples
        )

        total_samples = total_samples_from_file_resume(
            file_resume=self.paths.file_resume
        )

        log_evidence = log_evidence_from_file_summary(
            file_summary=self.paths.file_summary, prior_count=model.prior_count
        )

        return samples.NestedSamplerSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            total_samples=total_samples,
            log_evidence=log_evidence,
            number_live_points=self.n_live_points,
        )


def parameters_from_file_weighted_samples(
    file_weighted_samples, prior_count
) -> [[float]]:
    """Open the file "multinest.txt" and extract the parameter values of every accepted live point as a list
    of lists."""
    weighted_samples = open(file_weighted_samples)

    total_samples = 0
    for line in weighted_samples:
        total_samples += 1

    weighted_samples.seek(0)

    parameters = []

    for line in range(total_samples):
        vector = []
        weighted_samples.read(56)
        for param in range(prior_count):
            vector.append(float(weighted_samples.read(28)))
        weighted_samples.readline()
        parameters.append(vector)

    weighted_samples.close()

    return parameters


def log_likelihoods_from_file_weighted_samples(file_weighted_samples) -> [float]:
    """Open the file "multinest.txt" and extract the log likelihood values of every accepted live point as a list."""
    weighted_samples = open(file_weighted_samples)

    total_samples = 0
    for line in weighted_samples:
        total_samples += 1

    weighted_samples.seek(0)

    log_likelihoods = []

    for line in range(total_samples):
        weighted_samples.read(28)
        log_likelihoods.append(-0.5 * float(weighted_samples.read(28)))
        weighted_samples.readline()

    weighted_samples.close()

    return log_likelihoods


def weights_from_file_weighted_samples(file_weighted_samples) -> [float]:
    """Open the file "multinest.txt" and extract the weight values of every accepted live point as a list."""
    weighted_samples = open(file_weighted_samples)

    total_samples = 0
    for line in weighted_samples:
        total_samples += 1

    weighted_samples.seek(0)

    log_likelihoods = []

    for line in range(total_samples):
        weighted_samples.read(4)
        log_likelihoods.append(float(weighted_samples.read(24)))
        weighted_samples.readline()

    weighted_samples.close()

    return log_likelihoods


def total_samples_from_file_resume(file_resume):
    """Open the file "resume.dat" and extract the total number of samples of the MultiNest analysis
    (e.g. accepted + rejected)."""
    resume = open(file_resume)

    resume.seek(1)
    resume.read(19)
    total_samples = int(resume.read(8))
    resume.close()
    return total_samples


def log_evidence_from_file_summary(file_summary, prior_count):
    """Open the file "multinestsummary.txt" and extract the log evidence of the Multinest analysis.

    Early in the analysis this file may not yet have been created, in which case the log evidence estimate is
    unavailable and (would be unreliable anyway). In this case, a large negative value is returned."""

    try:

        with open(file_summary) as summary:

            summary.read(2 + 112 * prior_count)
            return float(summary.read(28))

    except FileNotFoundError:
        return -1.0e99
