import datetime as dt
import functools
import glob
import logging
import os
import shutil
import time

import numpy as np

from autofit import conf
from autofit.mapper import link, model_mapper as mm
from autofit.tools import path_util, text_util

logging.basicConfig()
logger = logging.getLogger(__name__)


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
            self.phase_tag = 'settings' + phase_tag

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

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.restore()

    def restore(self):
        """
        Copy files from the backup folder to the sym-linked optimizer folder.
        """
        if os.path.exists(self.backup_path):
            self.path = link.make_linked_folder(self.sym_path)
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
            os.makedirs(self.backup_path)
        except FileExistsError:
            pass

        self.create_paramnames_file()

        text_util.output_list_of_strings_to_file(file=self.file_model_info, list_of_strings=self.variable.info)

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

    def latex_results_at_sigma_limit(self, sigma_limit, format_str='{:.2f}'):

        labels = self.param_labels
        most_probables = self.most_probable_model_parameters
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)

        line = []

        for i in range(len(labels)):

            most_probable = format_str.format(most_probables[i])
            upper = format_str.format(uppers[i])
            lower = format_str.format(lowers[i])

            line += [labels[i] + ' = ' + most_probable + '^{+' + upper + '}_{-' + lower + '} & ']

        return line

    def create_paramnames_file(self):
        """The param_names file lists every parameter's analysis_path and Latex tag, and is used for *GetDist*
        visualization.

        The parameter names are determined from the class instance names of the model_mapper. Latex tags are
        properties of each model class."""
        paramnames_names = self.variable.param_names
        paramnames_labels = self.param_labels

        paramnames = []

        for i in range(self.variable.prior_count):
            line = text_util.label_and_label_string(label0=paramnames_names[i],
                                                    label1=paramnames_labels[i], whitespace=70)
            paramnames += [line + '\n']

        text_util.output_list_of_strings_to_file(file=self.file_param_names, list_of_strings=paramnames)

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

    @property
    def most_probable_model_parameters(self):
        raise NotImplementedError()

    @property
    def most_likely_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lensing.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        raise NotImplementedError()

    @property
    def maximum_likelihood(self):
        raise NotImplementedError()

    @property
    def maximum_log_likelihood(self):
        raise NotImplementedError()

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
        raise NotImplementedError()

    def model_parameters_at_upper_sigma_limit(self, sigma_limit):
        raise NotImplementedError()

    def model_parameters_at_lower_sigma_limit(self, sigma_limit):
        raise NotImplementedError

    @property
    def total_samples(self):
        raise NotImplementedError()

    def sample_model_parameters_from_sample_index(self, sample_index):
        raise NotImplementedError()

    @property
    def most_probable_model_instance(self):
        return self.variable.instance_from_physical_vector(physical_vector=self.most_probable_model_parameters)

    @property
    def most_likely_model_instance(self):
        return self.variable.instance_from_physical_vector(physical_vector=self.most_likely_model_parameters)

    def model_errors_at_sigma_limit(self, sigma_limit):
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)
        return list(map(lambda upper, lower: upper - lower, uppers, lowers))

    def model_errors_at_upper_sigma_limit(self, sigma_limit):
        uppers = self.model_parameters_at_upper_sigma_limit(sigma_limit=sigma_limit)
        return list(
            map(lambda upper, most_probable: upper - most_probable, uppers, self.most_probable_model_parameters))

    def model_errors_at_lower_sigma_limit(self, sigma_limit):
        lowers = self.model_parameters_at_lower_sigma_limit(sigma_limit=sigma_limit)
        return list(
            map(lambda lower, most_probable: most_probable - lower, lowers, self.most_probable_model_parameters))

    def sample_model_instance_from_sample_index(self, sample_index):
        """Setup a model instance of a weighted sample.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        model_parameters = self.sample_model_parameters_from_sample_index(sample_index=sample_index)

        return self.variable.instance_from_physical_vector(physical_vector=model_parameters)

    def sample_weight_from_sample_index(self, sample_index):
        raise NotImplementedError()

    def sample_likelihood_from_sample_index(self, sample_index):
        raise NotImplementedError()

    def offset_values_from_input_model_parameters(self, input_model_parameters):
        return list(map(lambda input, mp: mp - input, input_model_parameters, self.most_probable_model_parameters))


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
