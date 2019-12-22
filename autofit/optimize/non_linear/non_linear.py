import datetime as dt
import functools
import logging
import time

import numpy as np

from autofit import conf
from autofit.mapper import model_mapper as mm
from autofit.optimize.non_linear.paths import Paths, convert_paths

logging.basicConfig()
logger = logging.getLogger(__name__)  # TODO: Logging issue


class NonLinearOptimizer:
    @convert_paths
    def __init__(self, paths):
        """Abstract base class for non-linear optimizers.

        This class sets up the file structure for the non-linear optimizer nlo, which are standardized across \
        all non-linear optimizers.

        Parameters
        ------------

        """
        log_file = conf.instance.general.get("output", "log_file", str).replace(" ", "")
        self.paths = paths

        if not len(log_file) == 0:
            log_path = "{}/{}".format(self.paths.phase_output_path, log_file)
            logger.handlers = [logging.FileHandler(log_path)]
            logger.propagate = False
            # noinspection PyProtectedMember
            logger.level = logging._nameToLevel[
                conf.instance.general.get("output", "log_level", str)
                .replace(" ", "")
                .upper()
            ]

        self.paths.restore()

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
        return conf.instance.non_linear.get(
            self.__class__.__name__, attribute_name, attribute_type
        )

    def __eq__(self, other):
        return isinstance(other, NonLinearOptimizer) and self.__dict__ == other.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.paths.restore()

    def fit(self, analysis, model):
        raise NotImplementedError(
            "Fitness function must be overridden by non linear optimizers"
        )

    class Fitness:
        def __init__(
            self, paths, analysis, output_results=lambda during_analysis: None
        ):
            self.output_results = output_results
            self.paths = paths
            self.result = None
            self.max_likelihood = -np.inf
            self.analysis = analysis

            self.log_interval = conf.instance.general.get("output", "log_interval", int)
            self.backup_interval = conf.instance.general.get(
                "output", "backup_interval", int
            )
            self.visualize_interval = conf.instance.visualize.get(
                "figures", "visualize_interval", int
            )
            self.model_results_output_interval = conf.instance.general.get(
                "output", "model_results_output_interval", int
            )

            self.should_log = IntervalCounter(self.log_interval)
            self.should_backup = IntervalCounter(self.backup_interval)
            self.should_visualize = IntervalCounter(self.visualize_interval)
            self.should_output_model_results = IntervalCounter(
                self.model_results_output_interval
            )

        def fit_instance(self, instance):
            likelihood = self.analysis.fit(instance)

            if likelihood > self.max_likelihood:

                self.max_likelihood = likelihood
                self.result = Result(instance, likelihood)

                if self.should_visualize():
                    self.analysis.visualize(instance, during_analysis=True)

                if self.should_backup():
                    self.paths.backup()

                if self.should_output_model_results():
                    self.output_results(during_analysis=True)

            return likelihood

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        name = "{}/{}".format(self.paths.phase_name, extension)

        if remove_phase_tag:
            phase_tag = ""
        else:
            phase_tag = self.paths.phase_tag

        new_instance = self.__class__(
            Paths(
                phase_name=name,
                phase_folders=self.paths.phase_folders,
                phase_tag=phase_tag,
                remove_files=self.paths.remove_files,
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

    def __init__(
        self, instance, figure_of_merit, previous_model=None, gaussian_tuples=None
    ):
        """
        The result of an optimization.

        Parameters
        ----------
        instance: autofit.mapper.model.ModelInstance
            An instance object comprising the class instances that gave the optimal fit
        figure_of_merit: float
            A value indicating the figure of merit given by the optimal fit
        previous_model: mm.ModelMapper
            The model mapper from the stage that produced this result
        """
        self.instance = instance
        self.figure_of_merit = figure_of_merit
        self.previous_model = previous_model
        self.gaussian_tuples = gaussian_tuples
        self.__model = None

    @property
    def model(self):
        if self.__model is None:
            self.__model = self.previous_model.mapper_from_gaussian_tuples(
                self.gaussian_tuples
            )
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def __str__(self):
        return "Analysis Result:\n{}".format(
            "\n".join(
                ["{}: {}".format(key, value) for key, value in self.__dict__.items()]
            )
        )

    def model_absolute(self, a: float) -> mm.ModelMapper:
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
        return self.previous_model.mapper_from_gaussian_tuples(
            self.gaussian_tuples, a=a
        )

    def model_relative(self, r: float) -> mm.ModelMapper:
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
        return self.previous_model.mapper_from_gaussian_tuples(
            self.gaussian_tuples, r=r
        )


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
        start_time_path = "{}/.start_time".format(
            optimizer_instance.paths.phase_output_path
        )
        try:
            with open(start_time_path) as f:
                start = float(f.read())
        except FileNotFoundError:
            start = time.time()
            with open(start_time_path, "w+") as f:
                f.write(str(start))

        result = func(optimizer_instance, *args, **kwargs)

        execution_time = str(dt.timedelta(seconds=time.time() - start))

        logger.info(
            "{} took {} to run".format(
                optimizer_instance.paths.phase_name, execution_time
            )
        )
        with open(optimizer_instance.paths.execution_time_path, "w+") as f:
            f.write(execution_time)
        return result

    return timed_function
