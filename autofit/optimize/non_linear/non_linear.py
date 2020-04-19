import logging
import shutil
from abc import ABC, abstractmethod

import numpy as np

from autofit import conf
from autofit.plot import samples_text
from autofit.mapper import model_mapper as mm
from autofit.optimize.non_linear.paths import Paths, convert_paths
from autofit.tools import text_util

logging.basicConfig()
logger = logging.getLogger(__name__)  # TODO: Logging issue


class NonLinearOptimizer(ABC):
    @convert_paths
    def __init__(self, paths=None):
        """Abstract base class for non-linear optimizers.

        This class sets up the file structure for the non-linear optimizer nlo, which are standardized across \
        all non-linear optimizers.

        Parameters
        ------------

        """

        if paths is None:
            paths = Paths()

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

    @classmethod
    def simple_fit(
            cls,
            model,
            fitness_function,
            remove_output=True
    ) -> "Result":
        """
        Fit a model, M with some function f that takes instances of the
        class represented by model M and gives a score for their fitness

        Parameters
        ----------
        model
            A model that can be instantiated
        fitness_function
            A function that takes an instance of the model and scores it
        remove_output
            If True then output files are removed after the fit

        Returns
        -------
        A result comprising a score, the best fit instance and an updated prior model
        """
        optimizer = cls()

        result = optimizer._simple_fit(
            model,
            fitness_function
        )
        if remove_output:
            shutil.rmtree(
                optimizer.paths.phase_output_path
            )
        return result

    @abstractmethod
    def _simple_fit(self, model, fitness_function):
        pass

    @abstractmethod
    def _fit(self, analysis, model):
        pass

    def fit(
            self,
            analysis: "Analysis",
            model
    ) -> "Result":
        """
        A model which represents possible instances with some dimensionality is fit.

        The analysis provides two functions. One visualises an instance of a model and the
        other scores an instance based on how well it fits some data. The optimizer
        produces instances of the model by picking points in an N dimensional space.

        Parameters
        ----------
        analysis
            An object that encapsulates some data and implements a fit function
        model
            An object that represents possible instances of some model with a
            given dimensionality which is the number of free dimensions of the
            model.

        Returns
        -------
        An object encapsulating how well the model fit the data, the best fit instance
        and an updated model with free parameters updated to represent beliefs
        produced by this fit.
        """

        self.save_paramnames_file(model=model)

        result = self._fit(
            analysis,
            model
        )
        open(self.paths.has_completed_path, "w+").close()
        return result

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
        return conf.instance.non_linear.config_for(
            self.__class__.__name__).get(
            "general",
            attribute_name,
            attribute_type
        )

    def save_paramnames_file(self, model):
        """Create the param_names file listing every parameter's label and Latex tag, which is used for *GetDist*
        visualization.

        The parameter labels are determined using the label.ini and label_format.ini config files."""

        paramnames_names = model.param_names
        paramnames_labels = text_util.param_labels_from_model(model=model)

        paramnames = []

        for i in range(model.prior_count):
            line = text_util.label_and_label_string(
                label0=paramnames_names[i], label1=paramnames_labels[i], whitespace=70
            )
            paramnames += [line + "\n"]

        text_util.output_list_of_strings_to_file(
            file=self.paths.file_param_names, list_of_strings=paramnames
        )

    def __eq__(self, other):
        return isinstance(other, NonLinearOptimizer) and self.__dict__ == other.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.paths.restore()

    class Fitness:
        def __init__(
                self, paths, analysis, model, samples_from_model
        ):

            self.paths = paths
            self.result = None
            self.max_likelihood = -np.inf
            self.analysis = analysis

            self.model = model
            self.samples_from_model = samples_from_model

            self.log_interval = conf.instance.general.get("output", "log_interval", int)
            self.backup_interval = conf.instance.general.get(
                "output", "backup_interval", int
            )
            self.visualize_interval = conf.instance.visualize_general.get(
                "general", "visualize_interval", int
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
            log_likelihood = self.analysis.fit(instance)

            if log_likelihood > self.max_likelihood:

                self.max_likelihood = log_likelihood
                self.result = Result(instance, log_likelihood)

                if self.should_visualize():
                    self.analysis.visualize(instance, during_analysis=True)

                if self.should_backup():
                    self.paths.backup()

                if self.should_output_model_results():

                    samples = self.samples_from_model(model=self.model, paths=self.paths)
                    samples_text.output_results(samples=samples, during_analysis=True)

            return log_likelihood

        @property
        def samples(self):
            return self.samples_from_model(model=self.model, paths=self.paths)

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

    def samples_from_model(self, model, paths):
        raise NotImplementedError()


class Analysis:
    def fit(self, instance):
        raise NotImplementedError()

    def visualize(self, instance, during_analysis):
        raise NotImplementedError()


class Result:
    """
    @DynamicAttrs
    """

    def __init__(
            self, instance, log_likelihood, output=None, previous_model=None, gaussian_tuples=None
    ):
        """
        The result of an optimization.

        Parameters
        ----------
        instance: autofit.mapper.model.ModelInstance
            An instance object comprising the class instances that gave the optimal fit
        log_likelihood: float
            A value indicating the figure of merit given by the optimal fit
        previous_model
            The model mapper from the stage that produced this result
        """
        self.instance = instance
        self.log_likelihood = log_likelihood
        self.output = output
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


class IntervalCounter:
    def __init__(self, interval):
        self.count = 0
        self.interval = interval

    def __call__(self):
        if self.interval == -1:
            return False
        self.count += 1
        return self.count % self.interval == 0
