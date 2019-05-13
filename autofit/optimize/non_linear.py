import ast
import datetime as dt
import functools
import glob
import logging
import math
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pymultinest
import scipy.optimize

from autofit import conf
from autofit import exc
from autofit.mapper import model_mapper as mm, link
from autofit.optimize import optimizer as opt
from autofit.tools import path_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Analysis(object):

    def fit(self, instance):
        raise NotImplementedError()

    def visualize(self, instance, image_path, during_analysis):
        raise NotImplementedError()

    def describe(self, instance):
        raise NotImplementedError()


class Result(object):

    def __init__(self, constant, figure_of_merit, previous_variable=None, gaussian_tuples=None):
        """
        The result of an optimization.

        Parameters
        ----------
        constant: autofit.mapper.model.ModelInstance
            An instance object comprising the class instances that gave the optimal fit
        figure_of_merit: float
            A value indicating the figure of merit given by the optimal fit
        previous_variable: mm.ModelMapper
            The model mapper from the stage that produced this result
        """
        self.constant = constant
        self.figure_of_merit = figure_of_merit
        self.previous_variable = previous_variable
        self.gaussian_tuples = gaussian_tuples

    @property
    def variable(self) -> mm.ModelMapper:
        """
        A model mapper created by taking results from this phase and combining them with prior widths defined in the
        configuration.
        """
        return self.previous_variable.mapper_from_gaussian_tuples(self.gaussian_tuples)

    def __str__(self):
        return "Analysis Result:\n{}".format(
            "\n".join(["{}: {}".format(key, value) for key, value in self.__dict__.items()]))

    def variable_absolute(self, a: float) -> mm.ModelMapper:
        """
        Parameters
        ----------
        a
            The absolute width of gaussian priors

        Returns
        -------
        A model mapper created by taking results from this phase and creating priors with the defined absolute width.
        """
        return self.previous_variable.mapper_from_gaussian_tuples(self.gaussian_tuples, a=a)

    def variable_relative(self, r: float) -> mm.ModelMapper:
        """
        Parameters
        ----------
        r
            The relative width of gaussian priors

        Returns
        -------
        A model mapper created by taking results from this phase and creating priors with the defined relative width.
        """
        return self.previous_variable.mapper_from_gaussian_tuples(self.gaussian_tuples, r=r)


class IntervalCounter(object):
    def __init__(self, interval):
        self.count = 0
        self.interval = interval

    def __call__(self):
        if self.interval == -1:
            return False
        self.count += 1
        return self.count % self.interval == 0


def persistent_timer(func):
    """
    Times the execution of a function. If the process is stopped and restarted then timing is continued using saved
    files.

    Parameters
    ----------
    func
        Some function to be timed

    Returns
    -------
    timed_function
        The same function with a timer attached.
    """

    @functools.wraps(func)
    def timed_function(optimizer_instance, *args, **kwargs):
        start_time_path = "{}/.start_time".format(optimizer_instance.phase_output_path)
        try:
            with open(start_time_path) as f:
                start = float(f.read())
        except FileNotFoundError:
            start = time.time()
            with open(start_time_path, "w+") as f:
                f.write(str(start))

        result = func(optimizer_instance, *args, **kwargs)

        execution_time = str(dt.timedelta(seconds=time.time() - start))

        logger.info("{} took {} to run".format(
            optimizer_instance.phase_name,
            execution_time
        ))
        with open("{}/execution_time".format(optimizer_instance.phase_output_path), "w+") as f:
            f.write(execution_time)
        return result

    return timed_function


class NonLinearOptimizer(object):

    def __init__(self, phase_name, phase_tag=None, phase_folders=None, model_mapper=None):
        """Abstract base class for non-linear optimizers.

        This class sets up the file structure for the non-linear optimizer nlo, which are standardized across all \
        non-linear optimizers.

        Parameters
        ------------

        """
        self.named_config = conf.instance.non_linear

        self.phase_folders = phase_folders
        if phase_folders is None:
            self.phase_path = ''
        else:
            self.phase_path = path_util.path_from_folder_names(folder_names=phase_folders)

        self.phase_name = phase_name

        if phase_tag is None:
            self.phase_tag = ''
        else:
            self.phase_tag = phase_tag
            if len(self.phase_tag) > 1:
                if self.phase_tag[0] is '_':
                    self.phase_tag = self.phase_tag[1:]

        try:
            os.makedirs("/".join(self.sym_path.split("/")[:-1]))
        except FileExistsError:
            pass

        self.path = link.make_linked_folder(self.sym_path)

        self.variable = model_mapper or mm.ModelMapper()

        self.label_config = conf.instance.label

        self.log_file = conf.instance.general.get('output', 'log_file', str).replace(" ", "")

        if not len(self.log_file) == 0:
            log_path = "{}{}".format(self.phase_output_path, self.log_file)
            logger.handlers = [logging.FileHandler(log_path)]
            logger.propagate = False
            # noinspection PyProtectedMember
            logger.level = logging._nameToLevel[
                conf.instance.general.get('output', 'log_level', str).replace(" ", "").upper()]

        try:
            os.makedirs(self.image_path)
        except FileExistsError:
            pass

        try:
            os.makedirs("{}fits/".format(self.image_path))
        except FileExistsError:
            pass

        self.restore()

    @property
    def backup_path(self) -> str:
        """
        The path to the backed up optimizer folder.
        """
        return "{}/{}/{}/{}/optimizer_backup".format(conf.instance.output_path, self.phase_path, self.phase_name,
                                                     self.phase_tag)

    @property
    def phase_output_path(self) -> str:
        """
        The path to the output information for a phase.
        """
        return "{}/{}/{}/{}/".format(conf.instance.output_path, self.phase_path, self.phase_name,
                                     self.phase_tag)

    @property
    def opt_path(self) -> str:
        return "{}/{}/{}/{}/optimizer".format(conf.instance.output_path, self.phase_path, self.phase_name,
                                              self.phase_tag)

    @property
    def sym_path(self) -> str:
        return "{}/{}/{}/{}/optimizer".format(conf.instance.output_path, self.phase_path, self.phase_name,
                                              self.phase_tag)

    @property
    def file_param_names(self) -> str:
        return "{}/{}".format(self.opt_path, 'multinest.paramnames')

    @property
    def file_model_info(self) -> str:
        return "{}/{}".format(self.phase_output_path, 'model.info')

    @property
    def image_path(self) -> str:
        """
        The path to the directory in which images are stored.
        """
        return "{}image/".format(self.phase_output_path)

    def __eq__(self, other):
        return isinstance(other, NonLinearOptimizer) and self.__dict__ == other.__dict__

    def backup(self):
        """
        Copy files from the sym-linked optimizer folder to the backup folder in the workspace.
        """
        try:
            shutil.rmtree(self.backup_path)
        except FileNotFoundError:
            pass
        try:
            shutil.copytree(self.opt_path, self.backup_path)
        except shutil.Error as e:
            logger.exception(e)

    def restore(self):
        """
        Copy files from the backup folder to the sym-linked optimizer folder.
        """
        if os.path.exists(self.backup_path):
            for file in glob.glob(self.backup_path + "/*"):
                shutil.copy(file, self.path)

    def config(self, attribute_name, attribute_type=str):
        """
        Get a config field from this optimizer's section in non_linear.ini by a key and value type.

        Parameters
        ----------
        attribute_name: str
            The analysis_path of the field
        attribute_type: type
            The type of the value

        Returns
        -------
        attribute
            An attribute for the key with the specified type.
        """
        return self.named_config.get(self.__class__.__name__, attribute_name, attribute_type)

    def save_model_info(self):

        try:
            os.makedirs(self.path)
        except FileExistsError:
            pass

        self.create_paramnames_file()
        if not os.path.isfile(self.file_model_info):
            with open(self.file_model_info, 'w') as file:
                file.write(self.variable.info)
            file.close()

    def fit(self, analysis):
        raise NotImplementedError("Fitness function must be overridden by non linear optimizers")

    @property
    def param_labels(self):
        """The param_names vector is a list each parameter's analysis_path, and is used for *GetDist* visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are
        properties of each model class."""

        paramnames_labels = []
        prior_class_dict = self.variable.prior_class_dict
        prior_prior_model_dict = self.variable.prior_prior_model_dict

        for prior_name, prior in self.variable.prior_tuples_ordered_by_id:
            param_string = self.label_config.label(prior_name)
            prior_model = prior_prior_model_dict[prior]
            cls = prior_class_dict[prior]
            cls_string = "{}{}".format(self.label_config.subscript(cls), prior_model.component_number + 1)
            param_label = "{}_{{\\mathrm{{{}}}}}".format(param_string, cls_string)
            paramnames_labels.append(param_label)

        return paramnames_labels

    def create_paramnames_file(self):
        """The param_names file lists every parameter's analysis_path and Latex tag, and is used for *GetDist*
        visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are
        properties of each model class."""
        paramnames_names = self.variable.param_names
        paramnames_labels = self.param_labels
        with open(self.file_param_names, 'w') as paramnames:
            for i in range(self.variable.prior_count):
                line = paramnames_names[i]
                line += ' ' * (70 - len(line)) + paramnames_labels[i]
                paramnames.write(line + '\n')

    class Fitness(object):
        def __init__(self, nlo, analysis, image_path):
            self.nlo = nlo
            self.result = None
            self.max_likelihood = -np.inf
            self.image_path = image_path
            self.analysis = analysis
            visualise_interval = conf.instance.general.get('output', 'visualise_interval', int)
            log_interval = conf.instance.general.get('output', 'log_interval', int)
            backup_interval = conf.instance.general.get('output', 'backup_interval', int)

            self.should_log = IntervalCounter(log_interval)
            self.should_visualise = IntervalCounter(visualise_interval)
            self.should_backup = IntervalCounter(backup_interval)

        def fit_instance(self, instance):
            likelihood = self.analysis.fit(instance)

            if likelihood > self.max_likelihood:
                self.max_likelihood = likelihood
                self.result = Result(instance, likelihood)

                if self.should_visualise():
                    self.analysis.visualize(instance, image_path=self.image_path, during_analysis=True)

            if self.should_backup():
                self.nlo.backup()

            return likelihood

    def copy_with_name_extension(self, extension):
        name = "{}/{}".format(self.phase_name, extension)
        new_instance = self.__class__(phase_name=name, phase_folders=self.phase_folders, model_mapper=self.variable)
        return new_instance


class DownhillSimplex(NonLinearOptimizer):

    def __init__(self, phase_name, phase_tag=None, phase_folders=None, model_mapper=None, fmin=scipy.optimize.fmin):

        super(DownhillSimplex, self).__init__(phase_name=phase_name, phase_tag=phase_tag, phase_folders=phase_folders,
                                              model_mapper=model_mapper)

        self.xtol = self.config("xtol", float)
        self.ftol = self.config("ftol", float)
        self.maxiter = self.config("maxiter", int)
        self.maxfun = self.config("maxfun", int)

        self.full_output = self.config("full_output", int)
        self.disp = self.config("disp", int)
        self.retall = self.config("retall", int)

        self.fmin = fmin

        logger.debug("Creating DownhillSimplex NLO")

    def copy_with_name_extension(self, extension):
        copy = super().copy_with_name_extension(extension)
        copy.fmin = self.fmin
        copy.xtol = self.xtol
        copy.ftol = self.ftol
        copy.maxiter = self.maxiter
        copy.maxfun = self.maxfun
        copy.full_output = self.full_output
        copy.disp = self.disp
        copy.retall = self.retall
        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(self, nlo, analysis, instance_from_physical_vector, image_path):
            super().__init__(nlo, analysis, image_path)
            self.instance_from_physical_vector = instance_from_physical_vector

        def __call__(self, vector):
            try:
                instance = self.instance_from_physical_vector(vector)
                likelihood = self.fit_instance(instance)
            except exc.FitException:
                likelihood = -np.inf
            return -2 * likelihood

    @persistent_timer
    def fit(self, analysis):
        self.save_model_info()
        initial_vector = self.variable.physical_values_from_prior_medians

        fitness_function = DownhillSimplex.Fitness(self, analysis, self.variable.instance_from_physical_vector,
                                                   self.image_path)

        logger.info("Running DownhillSimplex...")
        output = self.fmin(fitness_function, x0=initial_vector)
        logger.info("DownhillSimplex complete")
        res = fitness_function.result

        # Create a set of Gaussian priors from this result and associate them with the result object.
        res.gaussian_tuples = [(mean, 0) for mean in output]
        res.previous_variable = self.variable

        analysis.visualize(instance=res.constant, image_path=self.image_path, during_analysis=False)

        self.backup()
        return res


class MultiNest(NonLinearOptimizer):

    def __init__(self, phase_name, phase_tag=None, phase_folders=None, model_mapper=None, sigma_limit=3,
                 run=pymultinest.run):
        """
        Class to setup and run a MultiNest lensing and output the MultiNest nlo.

        This interfaces with an input model_mapper, which is used for setting up the individual model instances that \
        are passed to each iteration of MultiNest.
        """

        super(MultiNest, self).__init__(phase_name=phase_name, phase_tag=phase_tag, phase_folders=phase_folders,
                                        model_mapper=model_mapper)

        self.file_summary = "{}/{}".format(self.path, 'multinestsummary.txt')
        self.file_weighted_samples = "{}/{}".format(self.path, 'multinest.txt')
        self.file_results = "{}/{}".format(self.phase_output_path, 'model.results')
        self._weighted_sample_model = None
        self.sigma_limit = sigma_limit

        self.importance_nested_sampling = self.config('importance_nested_sampling', bool)
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
        return getdist.mcsamples.loadMCSamples(self.opt_path + '/multinest')

    class Fitness(NonLinearOptimizer.Fitness):

        def __init__(self, nlo, analysis, instance_from_physical_vector, output_results, image_path):
            super().__init__(nlo, analysis, image_path)
            self.instance_from_physical_vector = instance_from_physical_vector
            self.output_results = output_results
            self.accepted_samples = 0
            self.number_of_accepted_samples_between_output = conf.instance.general.get(
                "output",
                "number_of_accepted_samples_between_output",
                int)

        def __call__(self, cube, ndim, nparams, lnew):
            try:
                instance = self.instance_from_physical_vector(cube)
                likelihood = self.fit_instance(instance)
            except exc.FitException:
                likelihood = -np.inf

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
            phys_cube = self.variable.physical_vector_from_hypercube_vector(hypercube_vector=cube)

            for i in range(self.variable.prior_count):
                cube[i] = phys_cube[i]

            return cube

        fitness_function = MultiNest.Fitness(self, analysis, self.variable.instance_from_physical_vector,
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

        self.output_results(during_analysis=False)
        self.output_pdf_plots()

        constant = self.most_likely_model_instance

        analysis.visualize(instance=constant, image_path=self.image_path, during_analysis=False)

        self.backup()
        return Result(constant=constant, figure_of_merit=self.maximum_likelihood,
                      previous_variable=self.variable,
                      gaussian_tuples=self.gaussian_priors_at_sigma_limit(self.sigma_limit))

    def open_summary_file(self):

        summary = open(self.file_summary)
        summary.seek(1)

        return summary

    def read_vector_from_summary(self, number_entries, offset):

        summary = self.open_summary_file()

        summary.seek(1)
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
        return self.read_vector_from_summary(number_entries=self.variable.prior_count, offset=0)

    @property
    def most_likely_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        return self.read_vector_from_summary(number_entries=self.variable.prior_count, offset=56)

    @property
    def maximum_likelihood(self):
        return self.read_vector_from_summary(number_entries=2, offset=112)[0]

    @property
    def maximum_log_likelihood(self):
        return self.read_vector_from_summary(number_entries=2, offset=112)[1]

    @property
    def most_probable_model_instance(self):
        return self.variable.instance_from_physical_vector(physical_vector=self.most_probable_model_parameters)

    @property
    def most_likely_model_instance(self):
        return self.variable.instance_from_physical_vector(physical_vector=self.most_likely_model_parameters)

    def gaussian_priors_at_sigma_limit(self, sigma_limit):
        """Compute the Gaussian Priors these results should be initialzed with in the next phase, by taking their \
        most probable values (e.g the means of their PDF) and computing the error at an input sigma_limit.

        Parameters
        -----------
        sigma_limit : float
            The sigma limit within which the PDF is used to estimate errors (e.g. sigma_limit = 1.0 uses 0.6826 of the \
            PDF).
        """

        means = self.most_probable_model_parameters
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)

        # noinspection PyArgumentList
        sigmas = list(map(lambda mean, upper, lower: max([upper - mean, mean - lower]), means, uppers, lowers))

        return list(map(lambda mean, sigma: (mean, sigma), means, sigmas))

    def model_parameters_at_sigma_limit(self, sigma_limit):
        limit = math.erf(0.5 * sigma_limit * math.sqrt(2))
        densities_1d = list(map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names))
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
        return list(map(lambda param: param[1], self.model_parameters_at_sigma_limit(sigma_limit)))

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
        return list(map(lambda param: param[0], self.model_parameters_at_sigma_limit(sigma_limit)))

    def model_errors_at_upper_sigma_limit(self, sigma_limit):
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        return list(
            map(lambda upper, most_probable: upper - most_probable, uppers, self.most_probable_model_parameters))

    def model_errors_at_lower_sigma_limit(self, sigma_limit):
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)
        return list(
            map(lambda lower, most_probable: most_probable - lower, lowers, self.most_probable_model_parameters))

    def model_errors_at_sigma_limit(self, sigma_limit):
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)
        return list(map(lambda upper, lower: upper - lower, uppers, lowers))

    def weighted_sample_model_instance_from_weighted_samples(self, index):
        """Setup a model instance of a weighted sample, including its weight and likelihood.

        Parameters
        -----------
        index : int
            The index of the weighted sample to return.
        """
        model, weight, likelihood = self.weighted_sample_model_parameters_from_weighted_samples(index)

        self._weighted_sample_model = model

        return self.variable.instance_from_physical_vector(model), weight, likelihood

    def weighted_sample_model_parameters_from_weighted_samples(self, index):
        """From a weighted sample return the model, weight and likelihood hood.

        NOTE: GetDist reads the log likelihood from the weighted_sample.txt file (column 2), which are defined as \
        -2.0*likelihood. This routine converts these back to likelihood.

        Parameters
        -----------
        index : int
            The index of the weighted sample to return.
        """
        return list(self.pdf.samples[index]), self.pdf.weights[index], -0.5 * self.pdf.loglikes[index]

    def offset_values_from_input_model_parameters(self, input_model_parameters):
        return list(map(lambda input, mp: mp - input, input_model_parameters, self.most_probable_model_parameters))

    def output_pdf_plots(self):

        import getdist.plots
        pdf_plot = getdist.plots.GetDistPlotter()

        plot_pdf_1d_params = conf.instance.general.get('output', 'plot_pdf_1d_params', bool)

        if plot_pdf_1d_params:

            for param_name in self.variable.param_names:
                pdf_plot.plot_1d(roots=self.pdf, param=param_name)
                pdf_plot.export(fname='{}/pdf_{}_1D.png'.format(self.image_path, param_name))

        plt.close()

        plot_pdf_triangle = conf.instance.general.get('output', 'plot_pdf_triangle', bool)

        if plot_pdf_triangle:

            try:
                pdf_plot.triangle_plot(roots=self.pdf)
                pdf_plot.export(fname='{}/pdf_triangle.png'.format(self.image_path))
            except Exception as e:
                print(type(e))
                print('The PDF triangle of this non-linear search could not be plotted. This is most likely due to a '
                      'lack of smoothness in the sampling of parameter space. Sampler further by decreasing the '
                      'parameter evidence_tolerance.')

        plt.close()

    def output_results(self, during_analysis=False):

        decimal_places = conf.instance.general.get("output", "model_results_decimal_places", int)

        def rounded(num):
            return np.round(num, decimal_places)

        if os.path.isfile(self.file_summary):

            with open(self.file_results, 'w') as results:

                results.write('Most likely model, Likelihood = {}\n'.format(rounded(self.maximum_likelihood)))
                results.write('\n')

                most_likely = self.most_likely_model_parameters

                if len(most_likely) != self.variable.prior_count:
                    raise exc.MultiNestException('MultiNest and GetDist have counted a different number of parameters.'
                                                 'See github issue https://github.com/Jammy2211/PyAutoLens/issues/49')

                for j in range(self.variable.prior_count):
                    most_likely_line = self.variable.param_names[j]
                    most_likely_line += ' ' * (60 - len(most_likely_line)) + str(rounded(most_likely[j]))
                    results.write(most_likely_line + '\n')

                if not during_analysis:

                    most_probable = self.most_probable_model_parameters

                    def write_for_sigma_limit(limit):
                        lower_limit = self.model_parameters_at_lower_sigma_limit(sigma_limit=limit)
                        upper_limit = self.model_parameters_at_upper_sigma_limit(sigma_limit=limit)

                        results.write('\n')
                        results.write('Most probable model ({} sigma limits)\n'.format(limit))
                        results.write('\n')

                        for i in range(self.variable.prior_count):
                            line = self.variable.param_names[i]
                            line += ' ' * (60 - len(line)) + str(
                                rounded(most_probable[i])) + ' (' + str(rounded(lower_limit[i])) + ', ' + str(
                                rounded(upper_limit[i])) + ')'
                            results.write(line + '\n')

                    write_for_sigma_limit(3.0)
                    write_for_sigma_limit(1.0)

                results.write('\n')
                results.write('Constants' + '\n')
                results.write('\n')

                constant_names = self.variable.constant_names
                constants = self.variable.constant_tuples_ordered_by_id

                for j in range(self.variable.constant_count):
                    constant_line = constant_names[j]
                    constant_line += ' ' * (60 - len(constant_line)) + str(constants[j][1].value)


class GridSearch(NonLinearOptimizer):

    def __init__(self, phase_name, phase_tag=None, phase_folders=None, step_size=None, model_mapper=None,
                 grid=opt.grid):
        """
        Optimise by performing a grid search.

        Parameters
        ----------
        step_size: float | None
            The step size of the grid search in hypercube space.
            E.g. a step size of 0.5 will give steps 0.0, 0.5 and 1.0
        model_mapper: cls
            The model mapper class (used for testing)
        phase_name: str
            The name of run (defaults to 'phase')
        grid: function
            A function that takes a fitness function, dimensionality and step size and performs a grid search
        """
        super().__init__(phase_name=phase_name, phase_tag=phase_tag, phase_folders=phase_folders,
                         model_mapper=model_mapper)
        self.step_size = step_size or self.config("step_size", float)
        self.grid = grid

    def copy_with_name_extension(self, extension):
        name = "{}/{}".format(self.phase_name, extension)
        new_instance = self.__class__(phase_name=name, phase_folders=self.phase_folders, model_mapper=self.variable,
                                      step_size=self.step_size)
        new_instance.grid = self.grid
        return new_instance

    class Result(Result):
        def __init__(self, result, instances, previous_variable, gaussian_tuples):
            """
            The result of an grid search optimization.

            Parameters
            ----------
            result: Result
                The result
            instances: [mm.ModelInstance]
                A model instance for each point in the grid search
            """
            super().__init__(result.constant, result.figure_of_merit, previous_variable, gaussian_tuples)
            self.instances = instances

        def __str__(self):
            return "Analysis Result:\n{}".format(
                "\n".join(["{}: {}".format(key, value) for key, value in self.__dict__.items()]))

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(self, nlo, analysis, instance_from_unit_vector, image_path, save_results,
                     checkpoint_count=0, best_fit=-np.inf, best_cube=None):
            super().__init__(nlo, analysis, image_path)
            self.instance_from_unit_vector = instance_from_unit_vector
            self.total_calls = 0
            self.checkpoint_count = checkpoint_count
            self.save_results = save_results
            self.best_fit = best_fit
            self.best_cube = best_cube
            self.all_fits = {}
            grid_results_interval = conf.instance.general.get('output', 'grid_results_interval', int)

            self.should_save_grid_results = IntervalCounter(grid_results_interval)
            if self.best_cube is not None:
                self.fit_instance(self.instance_from_unit_vector(self.best_cube))

        def __call__(self, cube):
            try:
                self.total_calls += 1
                if self.total_calls <= self.checkpoint_count:
                    #  TODO: is there an issue here where grid_search will forget the previous best fit?
                    return -np.inf
                instance = self.instance_from_unit_vector(cube)
                fit = self.fit_instance(instance)
                self.all_fits[cube] = fit
                if fit > self.best_fit:
                    self.best_fit = fit
                    self.best_cube = cube
                self.nlo.save_checkpoint(self.total_calls, self.best_fit, self.best_cube)
                if self.should_save_grid_results():
                    self.save_results(self.all_fits.items())
                return fit
            except exc.FitException:
                return -np.inf

    @property
    def checkpoint_path(self):
        return "{}/.checkpoint".format(self.path)

    def save_checkpoint(self, total_calls, best_fit, best_cube):
        with open(self.checkpoint_path, "w+") as f:
            def write(item):
                f.writelines("{}\n".format(item))

            write(total_calls)
            write(best_fit)
            write(best_cube)
            write(self.step_size)
            write(self.variable.prior_count)

    @property
    def is_checkpoint(self):
        return os.path.exists(self.checkpoint_path)

    @property
    def checkpoint_array(self):
        with open(self.checkpoint_path) as f:
            return f.readlines()

    @property
    def checkpoint_count(self):
        return int(self.checkpoint_array[0])

    @property
    def checkpoint_fit(self):
        return float(self.checkpoint_array[1])

    @property
    def checkpoint_cube(self):
        return ast.literal_eval(self.checkpoint_array[2])

    @property
    def checkpoint_step_size(self):
        return float(self.checkpoint_array[3])

    @property
    def checkpoint_prior_count(self):
        return int(self.checkpoint_array[4])

    @persistent_timer
    def fit(self, analysis):
        self.save_model_info()

        checkpoint_count = 0
        best_fit = -np.inf
        best_cube = None

        if self.is_checkpoint:
            if not self.checkpoint_prior_count == self.variable.prior_count:
                raise exc.CheckpointException("The number of dimensions does not match that found in the checkpoint")
            if not self.checkpoint_step_size == self.step_size:
                raise exc.CheckpointException("The step size does not match that found in the checkpoint")

            checkpoint_count = self.checkpoint_count
            best_fit = self.checkpoint_fit
            best_cube = self.checkpoint_cube

        def save_results(all_fit_items):
            results_list = [self.variable.param_names + ["fit"]]
            for item in all_fit_items:
                results_list.append([*self.variable.physical_vector_from_hypercube_vector(item[0]), item[1]])

            with open("{}/results".format(self.phase_output_path), "w+") as f:
                f.write("\n".join(map(lambda ls: ", ".join(
                    map(lambda value: "{:.2f}".format(value) if isinstance(value, float) else str(value), ls)),
                                      results_list)))

        fitness_function = GridSearch.Fitness(self,
                                              analysis,
                                              self.variable.instance_from_unit_vector,
                                              self.image_path,
                                              save_results,
                                              checkpoint_count=checkpoint_count,
                                              best_fit=best_fit,
                                              best_cube=best_cube)

        logger.info("Running grid search...")
        self.grid(fitness_function, self.variable.prior_count, self.step_size)

        logger.info("grid search complete")

        res = fitness_function.result

        instances = [(self.variable.instance_from_unit_vector(cube), fit) for cube, fit in
                     fitness_function.all_fits.items()]

        # Create a set of Gaussian priors from this result and associate them with the result object.
        res = GridSearch.Result(res, instances, self.variable, [(mean, 0) for mean in fitness_function.best_cube])

        analysis.visualize(instance=res.constant, image_path=self.image_path, during_analysis=False)

        self.backup()
        return res
