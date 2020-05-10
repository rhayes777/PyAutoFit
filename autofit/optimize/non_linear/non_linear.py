from abc import ABC, abstractmethod
import logging
import shutil
import numpy as np
import multiprocessing as mp
from time import sleep

from autofit import conf
from autofit.mapper import model_mapper as mm
from autofit.optimize.non_linear.paths import Paths, convert_paths
from autofit.text import formatter, samples_text

logging.basicConfig()
logger = logging.getLogger(__name__)  # TODO: Logging issue


class NonLinearOptimizer(ABC):
    @convert_paths
    def __init__(self, paths=None, number_of_cores=1):
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

        self.number_of_cores = number_of_cores

    @classmethod
    def fit(
            cls,
            model,
            analysis,
            remove_output=False
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

        result = optimizer._fit(
            model=model,
            analysis=analysis,
        )
        if remove_output:
            shutil.rmtree(
                optimizer.paths.phase_output_path
            )
        return result

    @abstractmethod
    def _fit(self, model, analysis):
        pass

    def full_fit(
            self,
            model,
            analysis: "Analysis",
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

        result = self._full_fit(
            model=model,
            analysis=analysis,
        )
        open(self.paths.has_completed_path, "w+").close()
        return result

    @abstractmethod
    def _full_fit(self, model, analysis):
        pass

    def config(self, section, attribute_name, attribute_type=str):
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
            section,
            attribute_name,
            attribute_type
        )

    def save_paramnames_file(self, model):
        """Create the param_names file listing every parameter's label and Latex tag, which is used for *GetDist*
        visualization.

        The parameter labels are determined using the label.ini and label_format.ini config files."""

        paramnames_names = model.param_names
        paramnames_labels = formatter.param_labels_from_model(model=model)

        paramnames = []

        for i in range(model.prior_count):
            line = formatter.label_and_label_string(
                label0=paramnames_names[i], label1=paramnames_labels[i], whitespace=70
            )
            paramnames += [f"{line}\n"]

        formatter.output_list_of_strings_to_file(
            file=self.paths.file_param_names, list_of_strings=paramnames
        )

    def __eq__(self, other):
        return isinstance(other, NonLinearOptimizer) and self.__dict__ == other.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.paths.restore()

    class Fitness:

        @staticmethod
        def prior(cube, model):

            # NEVER EVER REFACTOR THIS LINE! Haha.

            phys_cube = model.vector_from_unit_vector(unit_vector=cube)

            for i in range(len(phys_cube)):
                cube[i] = phys_cube[i]

            return cube

        @staticmethod
        def fitness(cube, model, fitness_function):
            return fitness_function(instance=model.instance_from_vector(cube))

        def __init__(
                self, paths, model, analysis, samples_from_model, pool_ids=None,
        ):

            self.paths = paths
            self.max_log_likelihood = -np.inf
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

            self.pool_ids = pool_ids

        def fit_instance(self, instance):

            log_likelihood = self.analysis.log_likelihood_function(instance=instance)

            if log_likelihood > self.max_log_likelihood:

                if self.pool_ids is not None:
                    if mp.current_process().pid != min(self.pool_ids):
                        return log_likelihood

                self.max_log_likelihood = log_likelihood

                if self.should_visualize():
                    self.analysis.visualize(instance, during_analysis=True)

                if self.should_backup():
                    self.paths.backup()

                if self.should_output_model_results():

                    try:
                        samples = self.samples_from_model(model=self.model)
                    except Exception:
                        samples = None

                    try:

                        samples_text.results_to_file(
                            samples=samples,
                            file_results=self.paths.file_results,
                            during_analysis=True
                        )

                    except (AttributeError, ValueError):
                        pass

            return log_likelihood

        @property
        def samples(self):
            return self.samples_from_model(model=self.model)

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        name = "{}/{}".format(self.paths.phase_name, extension)

        if remove_phase_tag:
            phase_tag = ""
        else:
            phase_tag = self.paths.phase_tag

        new_instance = self.__class__(
            paths=Paths(
                phase_name=name,
                phase_folders=self.paths.phase_folders,
                phase_tag=phase_tag,
                non_linear_name=self.paths.non_linear_name,
                remove_files=self.paths.remove_files,
            ),
        )

        return new_instance

    def samples_from_model(self, model):
        raise NotImplementedError()

    def make_pool(self):
        """Make the pool instance used to parallelize a non-linear search alongside a set of unique ids for every
        process in the pool. If the specified number of cores is 1, a pool instance is not made and None is returned.

        The pool cannot be set as an attribute of the class itself because this prevents pickling, thus it is generated
        via this function before calling the non-linear search.

        The pool instance is also set up with a list of unique pool ids, which are used during model-fitting to
        identify a 'master core' (the one whose id value is lowest) which handles model result output, visualization,
        etc."""

        if self.number_of_cores == 1:

            return None, None

        else:

            manager = mp.Manager()
            idQueue = manager.Queue()

            [idQueue.put(i) for i in range(self.number_of_cores)]

            pool = mp.Pool(processes=self.number_of_cores, initializer=init, initargs=(idQueue,))
            ids = pool.map(f, range(self.number_of_cores))

            return pool, [id[1] for id in ids]

class Analysis:
    def log_likelihood_function(self, instance):
        raise NotImplementedError()

    def visualize(self, instance, during_analysis):
        pass


class Result:
    """
    @DynamicAttrs
    """

    def __init__(
            self, samples, previous_model=None
    ):
        """
        The result of an optimization.

        Parameters
        ----------
            A value indicating the figure of merit given by the optimal fit
        previous_model
            The model mapper from the stage that produced this result
        """

        self.samples = samples

        self.previous_model = previous_model

        self.__model = None

        self._instance = samples.max_log_likelihood_instance if samples is not None else None

    @property
    def log_likelihood(self):
        return max(self.samples.log_likelihoods)

    @property
    def instance(self):
        return self._instance

    @property
    def max_log_likelihood_instance(self):
        return self._instance

    @property
    def model(self):
        if self.__model is None:
            self.__model = self.previous_model.mapper_from_gaussian_tuples(
                self.samples.gaussian_priors_at_sigma(sigma=3.0)
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
            self.samples.gaussian_priors_at_sigma(sigma=3.0), a=a
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
            self.samples.gaussian_priors_at_sigma(sigma=3.0), r=r
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


def init(queue):
    global idx
    idx = queue.get()

def f(x):
    global idx
    process = mp.current_process()
    sleep(1)
    return (idx, process.pid, x * x)