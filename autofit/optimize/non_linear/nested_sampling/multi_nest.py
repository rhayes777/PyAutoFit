import logging
import math
import os

import numpy as np
import pymultinest

from autofit import conf, exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.optimize.non_linear import samples
from autofit.optimize.non_linear.nested_sampling.nested_sampler import (
    NestedSampler,
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

        multinest_config = conf.instance.non_linear.config_for(
            "MultiNest"
        )
        self.terminate_at_acceptance_ratio = multinest_config.get(
            "general", "terminate_at_acceptance_ratio", bool
        )
        self.acceptance_ratio_threshold = multinest_config.get(
            "general", "acceptance_ratio_threshold", float
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
        of the full chains used by the fit.
        """

        def prior(cube, ndim, nparams):
            # NEVER EVER REFACTOR THIS LINE! Haha.

            phys_cube = model.vector_from_unit_vector(unit_vector=cube)

            for i in range(len(phys_cube)):
                cube[i] = phys_cube[i]

            return cube

        multinest = conf.instance.non_linear.config_for(
            "MultiNest"
        )
        stagger_resampling_likelihood = multinest.get(
            "general", "stagger_resampling_likelihood", bool
        )
        stagger_resampling_value = multinest.get(
            "general", "stagger_resampling_value", float
        )

        class Fitness:
            def __init__(self):
                """
                Fitness function that only handles resampling
                """
                self.stagger_accepted_samples = 0
                self.resampling_likelihood = multinest.get(
                    "general", "null_log_evidence", float
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
                        log_likelihood = -np.inf
                    else:
                        if self.stagger_accepted_samples < 10:
                            self.stagger_accepted_samples += 1
                            self.resampling_likelihood += stagger_resampling_value
                            log_likelihood = self.resampling_likelihood
                        else:
                            log_likelihood = (
                                -1.0 * np.abs(self.resampling_likelihood) * 10.0
                            )
                    return log_likelihood

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

        samples = self.samples_from_model(model=model, paths=self.paths)

        instance = samples.max_log_likelihood_instance
        samples.output_results(during_analysis=False)
        return Result(
            instance=instance,
            log_likelihood=samples.max_log_posterior,
            samples=samples,
            previous_model=model,
            gaussian_tuples=samples.gaussian_priors_at_sigma(self.sigma),
        )

    def samples_from_model(self, model, paths):
        """Create this non-linear search's output class from the model and paths.

        This function is required by the aggregator, so it knows which output class to generate an instance of."""
        return samples.NestedSamplerSamples(model=model, paths=paths)



