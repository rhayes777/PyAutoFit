import logging

import os
import numpy as np
import math
import pymultinest

from autofit import conf, exc
from autofit.optimize.non_linear.output import NestedSamplingOutput
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.non_linear import persistent_timer

logger = logging.getLogger(__name__)


class MultiNest(NonLinearOptimizer):
    def __init__(self, paths, sigma_limit=3, run=pymultinest.run):
        """
        Class to setup and run a MultiNest lens and output the MultiNest nlo.

        This interfaces with an input model_mapper, which is used for setting up the \
        individual model instances that are passed to each iteration of MultiNest.
        """

        super().__init__(paths)

        self.sigma_limit = sigma_limit

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
        self.outputfiles_basename = self.config("outputfiles_basename", str)
        self.seed = self.config("seed", int)
        self.verbose = self.config("verbose", bool)
        self.resume = self.config("resume", bool)
        self.context = self.config("context", int)
        self.write_output = self.config("write_output", bool)
        self.log_zero = self.config("log_zero", float)
        self.max_iter = self.config("max_iter", int)
        self.init_MPI = self.config("init_MPI", bool)
        self.run = run

        logger.debug("Creating MultiNest NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma_limit = self.sigma_limit
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
        copy.outputfiles_basename = self.outputfiles_basename
        copy.seed = self.seed
        copy.verbose = self.verbose
        copy.resume = self.resume
        copy.context = self.context
        copy.write_output = self.write_output
        copy.log_zero = self.log_zero
        copy.max_iter = self.max_iter
        copy.init_MPI = self.init_MPI
        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(
            self, paths, analysis, instance_from_physical_vector, output_results
        ):
            super().__init__(paths, analysis, output_results)
            self.instance_from_physical_vector = instance_from_physical_vector
            self.accepted_samples = 0

            self.model_results_output_interval = conf.instance.general.get(
                "output", "model_results_output_interval", int
            )
            self.stagger_resampling_likelihood = conf.instance.non_linear.get(
                "MultiNest", "stagger_resampling_likelihood", bool
            )
            self.stagger_resampling_value = conf.instance.non_linear.get(
                "MultiNest", "stagger_resampling_value", float
            )
            self.resampling_likelihood = conf.instance.non_linear.get(
                "MultiNest", "null_log_evidence", float
            )
            self.stagger_accepted_samples = 0

        def __call__(self, cube, ndim, nparams, lnew):

            try:
                instance = self.instance_from_physical_vector(cube)
                likelihood = self.fit_instance(instance)
            except exc.FitException:

                if not self.stagger_resampling_likelihood:
                    likelihood = -np.inf
                else:

                    if self.stagger_accepted_samples < 10:

                        self.stagger_accepted_samples += 1
                        self.resampling_likelihood += self.stagger_resampling_value
                        likelihood = self.resampling_likelihood

                    else:

                        likelihood = -1.0 * np.abs(self.resampling_likelihood) * 10.0

            return likelihood

    @persistent_timer
    def fit(self, analysis, model):
        multinest_output = MultiNestOutput(model, self.paths)

        multinest_output.save_model_info()

        # noinspection PyUnusedLocal
        def prior(cube, ndim, nparams):

            # NEVER EVER REFACTOR THIS LINE!

            phys_cube = model.physical_vector_from_hypercube_vector(
                hypercube_vector=cube
            )

            for i in range(len(phys_cube)):
                cube[i] = phys_cube[i]

            return cube

        fitness_function = MultiNest.Fitness(
            self.paths,
            analysis,
            model.instance_from_physical_vector,
            multinest_output.output_results,
        )

        logger.info("Running MultiNest...")
        self.run(
            fitness_function.__call__,
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
        logger.info("MultiNest complete")

        # TODO: Some of the results below use the backup_path, which isnt updated until the end if thiss function is
        # TODO: not located here. Do we need to rely just ono the optimizer foldeR? This is a good idea if we always
        # TODO: have a valid sym-link( e.g. even for aggregator use).

        self.paths.backup()
        instance = multinest_output.most_likely_model_instance
        analysis.visualize(instance=instance, during_analysis=False)
        multinest_output.output_results(during_analysis=False)
        multinest_output.output_pdf_plots()
        result = Result(
            instance=instance,
            figure_of_merit=multinest_output.evidence,
            previous_model=model,
            gaussian_tuples=multinest_output.gaussian_priors_at_sigma_limit(
                self.sigma_limit
            ),
        )
        self.paths.backup_zip_remove()
        return result


class MultiNestOutput(NestedSamplingOutput):
    @property
    def pdf(self):
        import getdist

        try:
            return getdist.mcsamples.loadMCSamples(
                self.paths.backup_path + "/multinest"
            )
        except IOError or OSError or ValueError or IndexError:
            raise Exception

    @property
    def pdf_converged(self):
        try:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            if densities_1d == []:
                return False

            return True
        except Exception:
            return False

    @property
    def most_probable_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lens.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.

        """
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=self.model.prior_count, offset=0
            )
        except FileNotFoundError:
            return self.most_likely_model_parameters

    @property
    def most_likely_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lens.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=self.model.prior_count, offset=56
            )
        except FileNotFoundError:
            most_likey_index = np.argmax([point[-1] for point in self.phys_live_points])
            return self.phys_live_points[most_likey_index][0:-1]

    @property
    def maximum_log_likelihood(self):
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=2, offset=112
            )[1]
        except FileNotFoundError:
            return max([point[-1] for point in self.phys_live_points])

    @property
    def evidence(self):
        try:
            return self.read_list_of_results_from_summary_file(
                number_entries=2, offset=112
            )[0]
        except FileNotFoundError:
            return None

    @property
    def phys_live_points(self):

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

    def read_list_of_results_from_summary_file(self, number_entries, offset):

        summary = open(self.paths.file_summary)
        summary.read(2 + offset * self.model.prior_count)
        vector = []
        for param in range(number_entries):
            vector.append(float(summary.read(28)))

        summary.close()

        return vector

    def model_parameters_at_sigma_limit(self, sigma_limit):
        limit = math.erf(0.5 * sigma_limit * math.sqrt(2))

        if self.pdf_converged:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            return list(map(lambda p: p.getLimits(limit), densities_1d))
        else:
            parameters_min = [
                min(
                    [
                        point[param_index]
                        for param_index in range(self.model.prior_count)
                    ]
                )
                for point in self.phys_live_points
            ]
            parameters_max = [
                max(
                    [
                        point[param_index]
                        for param_index in range(self.model.prior_count)
                    ]
                )
                for point in self.phys_live_points
            ]
            return [
                (parameters_min[index], parameters_max[index])
                for index in range(len(parameters_min))
            ]

    @property
    def total_samples(self):
        return len(self.pdf.weights)

    def sample_model_parameters_from_sample_index(self, sample_index):
        """From a sample return the model parameters.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return list(self.pdf.samples[sample_index])

    def sample_weight_from_sample_index(self, sample_index):
        """From a sample return the sample weight.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return self.pdf.weights[sample_index]

    def sample_likelihood_from_sample_index(self, sample_index):
        """From a sample return the likelihood.

        NOTE: GetDist reads the log likelihood from the weighted_sample.txt file (column 2), which are defined as \
        -2.0*likelihood. This routine converts these back to likelihood.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return -0.5 * self.pdf.loglikes[sample_index]
