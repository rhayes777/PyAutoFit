import math
import os

import numpy as np
import pymultinest
from matplotlib import pyplot as plt

from autofit import conf, exc
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer, persistent_timer, \
    Result
from autofit.optimize.non_linear.non_linear import logger
from autofit.tools import text_util


class MultiNest(NonLinearOptimizer):

    def __init__(self, phase_name, phase_tag=None, phase_folders=None,
                 model_mapper=None, sigma_limit=3,
                 run=pymultinest.run):
        """
        Class to setup and run a MultiNest lensing and output the MultiNest nlo.

        This interfaces with an input model_mapper, which is used for setting up the individual model instances that \
        are passed to each iteration of MultiNest.
        """

        super(MultiNest, self).__init__(phase_name=phase_name, phase_tag=phase_tag,
                                        phase_folders=phase_folders,
                                        model_mapper=model_mapper)

        self._weighted_sample_model = None
        self.sigma_limit = sigma_limit

        self.importance_nested_sampling = self.config('importance_nested_sampling',
                                                      bool)
        self.multimodal = self.config('multimodal', bool)
        self.const_efficiency_mode = self.config('const_efficiency_mode', bool)
        self.n_live_points = self.config('n_live_points', int)
        self.evidence_tolerance = self.config('evidence_tolerance', float)
        self.sampling_efficiency = self.config('sampling_efficiency', float)
        self.n_iter_before_update = self.config('n_iter_before_update', int)
        self.null_log_evidence = self.config('null_log_evidence', float)
        self.max_modes = self.config('max_modes', int)
        self.mode_tolerance = self.config('mode_tolerance', float)
        self.outputfiles_basename = self.config('outputfiles_basename', str)
        self.seed = self.config('seed', int)
        self.verbose = self.config('verbose', bool)
        self.resume = self.config('resume', bool)
        self.context = self.config('context', int)
        self.write_output = self.config('write_output', bool)
        self.log_zero = self.config('log_zero', float)
        self.max_iter = self.config('max_iter', int)
        self.init_MPI = self.config('init_MPI', bool)
        self.run = run

        logger.debug("Creating MultiNest NLO")

    @property
    def file_summary(self) -> str:
        return "{}/{}".format(self.backup_path, 'multinestsummary.txt')

    @property
    def file_weighted_samples(self):
        return "{}/{}".format(self.backup_path, 'multinest.txt')

    @property
    def file_results(self):
        return "{}/{}".format(self.phase_output_path, 'model.results')

    def copy_with_name_extension(self, extension):
        copy = super().copy_with_name_extension(extension)
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

    @property
    def pdf(self):
        import getdist
        return getdist.mcsamples.loadMCSamples(self.backup_path + '/multinest')

    class Fitness(NonLinearOptimizer.Fitness):

        def __init__(self, nlo, analysis, instance_from_physical_vector, output_results,
                     image_path):
            super().__init__(nlo, analysis, image_path)
            self.instance_from_physical_vector = instance_from_physical_vector
            self.output_results = output_results
            self.accepted_samples = 0
            self.number_of_accepted_samples_between_output = conf.instance.general.get(
                "output",
                "number_of_accepted_samples_between_output",
                int)
            self.stagger_resampling_likelihood = conf.instance.non_linear.get(
                'MultiNest', 'stagger_resampling_likelihood', bool)
            self.stagger_resampling_value = conf.instance.non_linear.get('MultiNest',
                                                                         'stagger_resampling_value',
                                                                         float)
            self.resampling_likelihood = conf.instance.non_linear.get('MultiNest',
                                                                      'null_log_evidence',
                                                                      float)

        def __call__(self, cube, ndim, nparams, lnew):
            try:
                instance = self.instance_from_physical_vector(cube)
                likelihood = self.fit_instance(instance)
            except exc.FitException:

                if not self.stagger_resampling_likelihood:
                    likelihood = -np.inf
                else:
                    self.resampling_likelihood += self.stagger_resampling_value
                    likelihood = self.resampling_likelihood

            if likelihood > self.max_likelihood:

                self.accepted_samples += 1

                if self.accepted_samples == self.number_of_accepted_samples_between_output:
                    self.accepted_samples = 0
                    self.output_results(during_analysis=True)

            return likelihood

    @persistent_timer
    def fit(self, analysis):

        self.save_model_info()

        # noinspection PyUnusedLocal
        def prior(cube, ndim, nparams):
            phys_cube = self.variable.physical_vector_from_hypercube_vector(
                hypercube_vector=cube)

            for i in range(self.variable.prior_count):
                cube[i] = phys_cube[i]

            return cube

        fitness_function = MultiNest.Fitness(self, analysis,
                                             self.variable.instance_from_physical_vector,
                                             self.output_results, self.image_path)

        logger.info("Running MultiNest...")
        self.run(fitness_function.__call__,
                 prior,
                 self.variable.prior_count,
                 outputfiles_basename="{}/multinest".format(self.path),
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
                 init_MPI=self.init_MPI)
        logger.info("MultiNest complete")

        self.backup()
        constant = self.most_likely_model_instance
        analysis.visualize(instance=constant, image_path=self.image_path,
                           during_analysis=False)
        self.output_results(during_analysis=False)
        self.output_pdf_plots()
        return Result(constant=constant, figure_of_merit=self.maximum_likelihood,
                      previous_variable=self.variable,
                      gaussian_tuples=self.gaussian_priors_at_sigma_limit(
                          self.sigma_limit))

    def read_list_of_results_from_summary_file(self, number_entries, offset):

        summary = open(self.file_summary)
        summary.read(2 + offset * self.variable.prior_count)
        vector = []
        for param in range(number_entries):
            vector.append(float(summary.read(28)))

        summary.close()

        return vector

    @property
    def most_probable_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.

        """
        return self.read_list_of_results_from_summary_file(
            number_entries=self.variable.prior_count, offset=0)

    @property
    def most_likely_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        return self.read_list_of_results_from_summary_file(
            number_entries=self.variable.prior_count, offset=56)

    @property
    def maximum_likelihood(self):
        return \
        self.read_list_of_results_from_summary_file(number_entries=2, offset=112)[0]

    @property
    def maximum_log_likelihood(self):
        return \
        self.read_list_of_results_from_summary_file(number_entries=2, offset=112)[1]

    def model_parameters_at_sigma_limit(self, sigma_limit):
        limit = math.erf(0.5 * sigma_limit * math.sqrt(2))
        densities_1d = list(
            map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names))
        return list(map(lambda p: p.getLimits(limit), densities_1d))

    def model_parameters_at_upper_sigma_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest nlo.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[1],
                        self.model_parameters_at_sigma_limit(sigma_limit)))

    def model_parameters_at_lower_sigma_limit(self, sigma_limit):
        """Setup 1D vectors of the upper and lower limits of the multinest nlo.

        These are generated at an input limfrac, which gives the percentage of 1d posterior weighted samples within \
        each parameter estimate

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """
        return list(map(lambda param: param[0],
                        self.model_parameters_at_sigma_limit(sigma_limit)))

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

    def output_pdf_plots(self):

        import getdist.plots
        pdf_plot = getdist.plots.GetDistPlotter()

        plot_pdf_1d_params = conf.instance.general.get('output', 'plot_pdf_1d_params',
                                                       bool)

        if plot_pdf_1d_params:

            for param_name in self.variable.param_names:
                pdf_plot.plot_1d(roots=self.pdf, param=param_name)
                pdf_plot.export(
                    fname='{}/pdf_{}_1D.png'.format(self.image_path, param_name))

        plt.close()

        plot_pdf_triangle = conf.instance.general.get('output', 'plot_pdf_triangle',
                                                      bool)

        if plot_pdf_triangle:

            try:
                pdf_plot.triangle_plot(roots=self.pdf)
                pdf_plot.export(fname='{}/pdf_triangle.png'.format(self.image_path))
            except Exception as e:
                print(type(e))
                print(
                    'The PDF triangle of this non-linear search could not be plotted. This is most likely due to a '
                    'lack of smoothness in the sampling of parameter space. Sampler further by decreasing the '
                    'parameter evidence_tolerance.')

        plt.close()

    def output_results(self, during_analysis=False):

        decimal_places = conf.instance.general.get("output",
                                                   "model_results_decimal_places", int)

        format_str = '{:.' + str(decimal_places) + 'f}'

        if os.path.isfile(self.file_summary):

            results = []

            likelihood = '{:.8f}'.format(self.maximum_likelihood)
            results += ['Most likely model, Likelihood = {}\n\n'.format(likelihood)]

            most_likely = self.most_likely_model_parameters

            if len(most_likely) != self.variable.prior_count:
                raise exc.MultiNestException(
                    'MultiNest and GetDist have counted a different number of parameters.'
                    'See github issue https://github.com/Jammy2211/PyAutoLens/issues/49')

            for j in range(self.variable.prior_count):
                line = text_util.label_and_value_string(
                    label=self.variable.param_names[j], value=most_likely[j],
                    whitespace=60, format_str=format_str)
                results += [line + '\n']

            if not during_analysis:

                most_probable_params = self.most_probable_model_parameters

                def results_from_sigma_limit(limit):

                    lower_limits = self.model_parameters_at_lower_sigma_limit(
                        sigma_limit=limit)
                    upper_limits = self.model_parameters_at_upper_sigma_limit(
                        sigma_limit=limit)

                    results = [
                        '\n\nMost probable model ({} sigma limits)\n\n'.format(limit)]

                    for i in range(self.variable.prior_count):
                        line = text_util.label_value_and_limits_string(
                            label=self.variable.param_names[i],
                            value=most_probable_params[i],
                            lower_limit=lower_limits[i],
                            upper_limit=upper_limits[i], whitespace=60,
                            format_str=format_str)

                        results += [line + '\n']

                    return results

                results += results_from_sigma_limit(limit=3.0)
                results += results_from_sigma_limit(limit=1.0)

            results += ['\n\nConstants\n\n']

            constant_names = self.variable.constant_names
            constants = self.variable.constant_tuples

            for j in range(self.variable.constant_count):
                line = text_util.label_and_value_string(label=constant_names[j],
                                                        value=constants[j][1].value,
                                                        whitespace=60,
                                                        format_str=format_str)
                results += [line + '\n']

            text_util.output_list_of_strings_to_file(file=self.file_results,
                                                     list_of_strings=results)
