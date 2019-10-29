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

logging.basicConfig()
logger = logging.getLogger(__name__)  # TODO: Logging issue


class Paths:
    def __init__(
            self,
            phase_name="",
            phase_tag=None,
            phase_folders=tuple()
    ):
        self.phase_path = "/".join(phase_folders)
        self.phase_name = phase_name
        self.phase_tag = phase_tag or ''

        try:
            os.makedirs("/".join(self.sym_path.split("/")[:-1]))
        except FileExistsError:
            pass

        try:
            os.makedirs(self.pdf_path)
        except FileExistsError:
            pass

        self.path = link.make_linked_folder(self.sym_path)

    def __eq__(self, other):
        return isinstance(other, Paths) and all([
            self.phase_path == other.phase_path,
            self.phase_name == other.phase_name,
            self.phase_tag == other.phase_tag
        ])

    @property
    def phase_folders(self):
        return self.phase_path.split("/")

    @property
    def backup_path(self) -> str:
        """
        The path to the backed up optimizer folder.
        """
        return "/".join(
            filter(
                lambda item: len(item) > 0,
                [
                    conf.instance.output_path,
                    self.phase_path,
                    self.phase_name,
                    self.phase_tag,
                    'optimizer_backup'
                ]
            )
        )

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

    @property
    def pdf_path(self) -> str:
        """
        The path to the directory in which images are stored.
        """
        return "{}pdf/".format(self.image_path)

    def make_optimizer_pickle_path(self) -> str:
        """
        Create the path at which the optimizer pickle should be saved
        """
        return "{}/optimizer.pickle".format(self.make_path())

    def make_model_pickle_path(self):
        """
        Create the path at which the model pickle should be saved
        """
        return "{}/model.pickle".format(self.make_path())

    def make_path(self) -> str:
        """
        Create the path to the folder at which the metadata and optimizer pickle should
        be saved
        """
        return "{}/{}/{}/{}/".format(
            conf.instance.output_path,
            self.phase_path,
            self.phase_name,
            self.phase_tag
        )

    @property
    def file_summary(self) -> str:
        return "{}/{}".format(self.backup_path, 'multinestsummary.txt')

    @property
    def file_weighted_samples(self):
        return "{}/{}".format(self.backup_path, 'multinest.txt')

    @property
    def file_results(self):
        return "{}/{}".format(self.phase_output_path, 'model.results')


class NonLinearOptimizer(object):

    def __init__(self, paths):
        """Abstract base class for non-linear optimizers.

        This class sets up the file structure for the non-linear optimizer nlo, which are standardized across \
        all non-linear optimizers.

        Parameters
        ------------

        """
        log_file = conf.instance.general.get('output', 'log_file', str).replace(" ", "")
        self.paths = paths

        if not len(log_file) == 0:
            log_path = "{}{}".format(
                self.paths.phase_output_path,
                log_file
            )
            logger.handlers = [logging.FileHandler(log_path)]
            logger.propagate = False
            # noinspection PyProtectedMember
            logger.level = logging._nameToLevel[
                conf.instance.general.get('output', 'log_level', str).replace(" ", "").upper()]

        self.restore()

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
        return conf.instance.non_linear.get(self.__class__.__name__, attribute_name, attribute_type)

    def __eq__(self, other):
        return isinstance(other, NonLinearOptimizer) and self.__dict__ == other.__dict__

    def backup(self):
        """
        Copy files from the sym-linked optimizer folder to the backup folder in the workspace.
        """

        try:
            shutil.rmtree(self.paths.backup_path)
        except FileNotFoundError:
            pass

        try:
            shutil.copytree(self.paths.opt_path, self.paths.backup_path)
        except shutil.Error as e:
            logger.exception(e)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.restore()

    def restore(self):
        """
        Copy files from the backup folder to the sym-linked optimizer folder.
        """
        if os.path.exists(self.paths.backup_path):
            self.paths.path = link.make_linked_folder(self.paths.sym_path)
            for file in glob.glob(self.paths.backup_path + "/*"):
                shutil.copy(file, self.paths.path)

    def fit(
            self,
            analysis,
            model
    ):
        raise NotImplementedError("Fitness function must be overridden by non linear optimizers")

    class Fitness:

        def __init__(self, nlo, analysis):

            self.nlo = nlo
            self.result = None
            self.max_likelihood = -np.inf
            self.analysis = analysis

            log_interval = conf.instance.general.get('output', 'log_interval', int)
            backup_interval = conf.instance.general.get('output', 'backup_interval', int)
            visualize_interval = conf.instance.visualize.get('figures', 'visualize_interval', int)

            self.should_log = IntervalCounter(log_interval)
            self.should_backup = IntervalCounter(backup_interval)
            self.should_visualize = IntervalCounter(visualize_interval)

        def fit_instance(self, instance):
            likelihood = self.analysis.fit(instance)

            if likelihood > self.max_likelihood:

                self.max_likelihood = likelihood
                self.result = Result(instance, likelihood)

                if self.should_visualize():
                    self.analysis.visualize(instance, during_analysis=True)

                if self.should_backup():
                    self.nlo.backup()

            return likelihood

    def copy_with_name_extension(self, extension):
        name = "{}/{}".format(self.paths.phase_name, extension)

        new_instance = self.__class__(
            Paths(
                phase_name=name,
                phase_folders=self.paths.phase_folders,
                phase_tag=self.paths.phase_tag
            )
        )

        return new_instance


class Analysis(object):

    def fit(self, instance):
        raise NotImplementedError()

    def visualize(self, instance, during_analysis):
        raise NotImplementedError()


class Result(object):
    """
    @DynamicAttrs
    """

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
        self.__variable = None

    @property
    def variable(self):
        if self.__variable is None:
            self.__variable = self.previous_variable.mapper_from_gaussian_tuples(
                self.gaussian_tuples
            )
        return self.__variable

    @variable.setter
    def variable(self, variable):
        self.__variable = variable

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
        A model mapper created by taking results from this phase and creating priors with the defined absolute
        width.
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
        A model mapper created by taking results from this phase and creating priors with the defined relative
        width.
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
    Times the execution of a function. If the process is stopped and restarted then timing is continued using
    saved files.

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
        start_time_path = "{}/.start_time".format(optimizer_instance.paths.phase_output_path)
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
            optimizer_instance.paths.phase_name,
            execution_time
        ))
        with open("{}/execution_time".format(optimizer_instance.paths.phase_output_path), "w+") as f:
            f.write(execution_time)
        return result

    return timed_function
