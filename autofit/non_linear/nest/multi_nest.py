from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import NestSamples
from autofit.non_linear.nest import abstract_nest
from autofit.non_linear import abstract_search

from autofit.non_linear.log import logger


class MultiNest(abstract_nest.AbstractNest):
    def __init__(
        self,
        paths=None,
        prior_passer=None,
        n_live_points=None,
        sampling_efficiency=None,
        const_efficiency_mode=None,
        multimodal=None,
        importance_nested_sampling=None,
        evidence_tolerance=None,
        max_modes=None,
        mode_tolerance=None,
        max_iter=None,
        n_iter_before_update=None,
        null_log_evidence=None,
        seed=None,
        verbose=None,
        resume=None,
        context=None,
        write_output=None,
        log_zero=None,
        init_MPI=None,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
        stagger_resampling_likelihood=None,
    ):
        """
        A MultiNest non-linear search.

        For a full description of MultiNest and its Python wrapper PyMultiNest, checkout its Github and documentation
        webpages:

        https://github.com/JohannesBuchner/MultiNest
        https://github.com/JohannesBuchner/PyMultiNest
        http://johannesbuchner.github.io/PyMultiNest/index.html#

        Parameters
        ----------
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this non-linear search to a subsequent non-linear search.
        n_live_points : int
            The number of live points used to sample non-linear parameter space. More points provides a more thorough
            sampling of parameter space, at the expense of taking longer to run. The number of live points required for
            accurate sampling depends on the complexity of parameter space.
        sampling_efficiency : float
            The ratio of accepted to total samples MultiNest targets. A higher efficiency will converges on the high
            log_likelihood regions of parameter space faster at the risk of missing the global maxima solution. By
            default we recommend a value of 0.8 (without constant efficiency mode) and 0.3 (with constant efficiency
            mode). Reduce to lower values if the inferred solution does not look accurate.
        const_efficiency_mode : bool
            The sampling efficiency determines the acceptance rate MultiNest targets. However, if MultiNest cannot map
            out parameter-space accurately it reduce the acceptance rate. Constant efficiency mode forces MultiNest to
            maintain the sampling efficiency acceptance rate. This can dramatically reduce run-times but increases the
            risk of missing the global maximum log likelihood solution.
        multimodal : bool
            Whether MultiNest uses multi-modal sampling, whereby the parameter space search will 'split' into
            multiple modes if it detects there are multiple peaks in log_likelihood space.
        importance_nested_sampling : bool
            Importance nested sampling mode uses information from the rejected points to improve the non-linear search.
        evidence_tolerance : float
            MultiNest will stop sampling when it estimates that continuing sampling will not increase the log evidence
            more than the evidence_tolerance value. Thus, the higher the evidence_tolerance the sooner MultiNest will
            stop running. Higher tolerances provide more accurate parameter errors.
        max_modes : int
            If multimodal sampling is True, the maximum number of models MultiNest can split into.
        mode_tolerance : float
            MultiNest can find multiple modes & also specify which samples belong to which mode. It might be desirable
            to have separate samples & mode statistics for modes with local log-evidence value greater than a
            particular value in which case Ztol should be set to that value. If there isn’t any particularly
            interesting Ztol value, then Ztol should be set to a very large negative number (e.g. -1e90).
        max_iter : int
            maximum number of iterations. 0 is unlimited.
        n_iter_before_update : int
            Number of accepted samples (times 10) per MultiNest output to hard disk.
        null_log_evidence : float
            If multimodal is True, MultiNest can find multiple modes & also specify which samples belong to which mode.
            It might be desirable to have separate samples & mode statistics for modes with local log-evidence value
            greater than a particular value in which case nullZ should be set to that value. If there isn’t any
            particulrly interesting nullZ value, then nullZ should be set to a very large negative number (e.g. -1.d90).
        seed : int
            The random number generator seed of MultiNest, enabling reproducible results.
        verbose : bool
            Whether MultiNest prints messages.
        resume : bool
            If True and existing results are found at the output path, MultiNest will resume that run. If False,
            MultiNest will start a new run.
        context : None
            Not used by PyAutoFit.
        write_output : bool
            Whether the results are written to the hard-disk as text files (allowing the run to be resumed).
        log_zero : float
            points with loglike < logZero will be ignored by MultiNest.
        init_MPI : None
            MPI not supported by PyAutoFit for MultiNest.
        terminate_at_acceptance_ratio : bool
            If *True*, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value (see *Nest* for a full description of this feature).
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is
            *True* (see *Nest* for a full description of this feature).
        """

        self.n_live_points = (
            self._config("search", "n_live_points", int)
            if n_live_points is None
            else n_live_points
        )
        self.sampling_efficiency = (
            self._config("search", "sampling_efficiency", float)
            if sampling_efficiency is None
            else sampling_efficiency
        )
        self.const_efficiency_mode = (
            self._config("search", "const_efficiency_mode", bool)
            if const_efficiency_mode is None
            else const_efficiency_mode
        )
        self.evidence_tolerance = (
            self._config("search", "evidence_tolerance", float)
            if evidence_tolerance is None
            else evidence_tolerance
        )

        if self.evidence_tolerance <= 0.0:
            self.evidence_tolerance = 0.8

        self.multimodal = (
            multimodal or self._config("search", "multimodal", bool)
            if multimodal is None
            else multimodal
        )
        self.importance_nested_sampling = (
            self._config("search", "importance_nested_sampling", bool)
            if importance_nested_sampling is None
            else importance_nested_sampling
        )
        self.max_modes = (
            self._config("search", "max_modes", int) if max_modes is None else max_modes
        )
        self.mode_tolerance = (
            self._config("search", "mode_tolerance", float)
            if mode_tolerance is None
            else mode_tolerance
        )
        self.max_iter = (
            self._config("search", "max_iter", int) if max_iter is None else max_iter
        )
        self.n_iter_before_update = (
            self._config("settings", "n_iter_before_update", int)
            if n_iter_before_update is None
            else n_iter_before_update
        )
        self.null_log_evidence = (
            self._config("settings", "null_log_evidence", float)
            if null_log_evidence is None
            else null_log_evidence
        )
        self.seed = self._config("settings", "seed", int) if seed is None else seed
        self.verbose = (
            self._config("settings", "verbose", bool) if verbose is None else verbose
        )
        self.resume = (
            self._config("settings", "resume", bool) if resume is None else resume
        )
        self.context = (
            self._config("settings", "context", int) if context is None else context
        )
        self.write_output = (
            self._config("settings", "write_output", bool)
            if write_output is None
            else write_output
        )
        self.log_zero = (
            self._config("settings", "log_zero", float) if log_zero is None else log_zero
        )
        self.init_MPI = (
            self._config("settings", "init_MPI", bool) if init_MPI is None else init_MPI
        )

        super().__init__(
            paths=paths,
            prior_passer=prior_passer,
            terminate_at_acceptance_ratio=terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=acceptance_ratio_threshold,
            stagger_resampling_likelihood=stagger_resampling_likelihood,
        )

        logger.debug("Creating MultiNest NLO")

    def _fit(self, model: AbstractPriorModel, analysis) -> abstract_search.Result:
        """
        Fit a model using MultiNest and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the non-linear search seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the non-linear search maximizes.

        Returns
        -------
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted ssamples of the fit.
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

        logger.info("Beginning MultiNest non-linear search. ")

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
            verbose=not self.silence,
            resume=self.resume,
            context=self.context,
            write_output=self.write_output,
            log_zero=self.log_zero,
            max_iter=self.max_iter,
            init_MPI=self.init_MPI,
        )

    @property
    def tag(self):
        """Tag the output folder of the PySwarms non-linear search, according to the number of particles and
        parameters defining the search strategy."""

        name_tag = self._config("tag", "name")
        n_live_points_tag = (
            f"{self._config('tag', 'n_live_points')}_{self.n_live_points}"
        )
        sampling_efficiency_tag = (
            f"{self._config('tag', 'sampling_efficiency')}_{self.sampling_efficiency}"
        )
        if self.const_efficiency_mode:
            const_efficiency_mode_tag = (
                f"_{self._config('tag', 'const_efficiency_mode')}"
            )
        else:
            const_efficiency_mode_tag = ""
        if self.multimodal:
            multimodal_tag = f"_{self._config('tag', 'multimodal')}"
        else:
            multimodal_tag = ""
        if self.importance_nested_sampling:
            importance_nested_sampling_tag = (
                f"_{self._config('tag', 'importance_nested_sampling')}"
            )
        else:
            importance_nested_sampling_tag = ""

        return f"{name_tag}__{n_live_points_tag}_{sampling_efficiency_tag}{const_efficiency_mode_tag}{multimodal_tag}{importance_nested_sampling_tag}"

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the multinest non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.prior_passer = self.prior_passer
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
        copy.stagger_resampling_likelihood = self.stagger_resampling_likelihood
        return copy

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

        return NestSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            total_samples=total_samples,
            log_evidence=log_evidence,
            number_live_points=self.n_live_points,
            time=self.timer.time
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
