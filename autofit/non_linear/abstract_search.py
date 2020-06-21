import configparser
import logging
import multiprocessing as mp
import pickle
import os
from abc import ABC, abstractmethod
from time import sleep
from typing import Dict

import numpy as np

from autoconf import conf
from autofit.mapper import model_mapper as mm
from autofit.non_linear.paths import Paths, convert_paths
from autofit.text import formatter
from autofit.text import model_text
from autofit.text import samples_text
from autofit import exc

logging.basicConfig()
logger = logging.getLogger(__name__)  # TODO: Logging issue


class NonLinearSearch(ABC):
    @convert_paths
    def __init__(
            self,
            paths=None,
            initialize_method=None,
            initialize_ball_lower_limit=None,
            initialize_ball_upper_limit=None,
            iterations_per_update=None,
            number_of_cores=1
    ):
        """Abstract base class for non-linear searches.

        This class sets up the file structure for the non-linear search, which are standardized across all non-linear
        searches.

        Parameters
        ------------
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search samples,
            backups, etc.
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        initialize_method : str
            The method used to generate where walkers are initialized in parameter space, with options:
            ball (default):
                Walkers are initialized by randomly drawing unit values from a uniform distribution between the
                initialize_ball_lower_limit and initialize_ball_upper_limit values. It is recommended these limits are
                small, such that all walkers begin close to one another.
            prior:
                Walkers are initialized by randomly drawing unit values from a uniform distribution between 0 and 1,
                thus being distributed over the prior.
        initialize_ball_lower_limit : float
            The lower limit of the uniform distribution unit values are drawn from when initializing walkers using the
            ball method.
        initialize_ball_upper_limit : float
            The upper limit of the uniform distribution unit values are drawn from when initializing walkers using the
            ball method.
        """

        if paths.non_linear_name is "":
            paths.non_linear_name = self.config("tag", "name")

        if paths.non_linear_tag is "":
            paths.non_linear_tag_function = lambda: self.tag

        self.paths = paths

        self.log_file = conf.instance.general.get("output", "log_file", str).replace(" ", "")

        try:

            self.initialize_method = (
                self.config("initialize", "method", str)
                if initialize_method is None
                else initialize_method
            )

        except configparser.NoSectionError:

            self.initialize_method = None

        try:

            self.initialize_ball_lower_limit = (
                self.config("initialize", "ball_lower_limit", float)
                if initialize_ball_lower_limit is None
                else initialize_ball_lower_limit
            )

        except configparser.NoSectionError:

            self.initialize_ball_lower_limit = None

        try:

            self.initialize_ball_upper_limit = (
                self.config("initialize", "ball_upper_limit", float)
                if initialize_ball_upper_limit is None
                else initialize_ball_upper_limit
            )

        except configparser.NoSectionError:

            self.initialize_ball_upper_limit = None

        self.iterations_per_update = (
            self.config("updates", "iterations_per_update", int)
            if iterations_per_update is None
            else iterations_per_update
        )

        self.log_every_update = self.config("updates", "log_every_update", int)
        self.backup_every_update = self.config(
            "updates", "backup_every_update", int
        )
        self.visualize_every_update = self.config(
            "updates", "visualize_every_update", int
        )
        self.model_results_every_update = self.config(
            "updates", "model_results_every_update", int
        )

        self.should_log = IntervalCounter(self.log_every_update)
        self.should_backup = IntervalCounter(self.backup_every_update)
        self.should_visualize = IntervalCounter(self.visualize_every_update)
        self.should_output_model_results = IntervalCounter(
            self.model_results_every_update
        )

        self.silence = self.config("printing", "silence", bool)

        self.number_of_cores = number_of_cores

        self._in_phase = False

    class Fitness:

        def __init__(
                self, paths, model, analysis, samples_from_model, pool_ids=None,
        ):

            self.paths = paths
            self.max_log_likelihood = -np.inf
            self.analysis = analysis

            self.model = model
            self.samples_from_model = samples_from_model

            self.pool_ids = pool_ids

        def fit_instance(self, instance):

            log_likelihood = self.analysis.log_likelihood_function(instance=instance)

            if self.analysis.log_likelihood_cap is not None:
                if log_likelihood > self.analysis.log_likelihood_cap:
                    log_likelihood = self.analysis.log_likelihood_cap

            if log_likelihood > self.max_log_likelihood:

                if self.pool_ids is not None:
                    if mp.current_process().pid != min(self.pool_ids):
                        return log_likelihood

                self.max_log_likelihood = log_likelihood

            return log_likelihood

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

        @property
        def samples(self):
            return self.samples_from_model(model=self.model)

        @property
        def resample_likelihood(self):
            """If a sample raises a FitException, this value is returned to signify that the point requires resampling or
             should be given a likelihood so low that it is discard."""
            return -np.inf

    def fit(
            self,
            model,
            analysis: "Analysis",
            info=None,
    ) -> "Result":
        """ Fit a model, M with some function f that takes instances of the
        class represented by model M and gives a score for their fitness.

        A model which represents possible instances with some dimensionality is fit.

        The analysis provides two functions. One visualises an instance of a model and the
        other scores an instance based on how well it fits some data. The search
        produces instances of the model by picking points in an N dimensional space.

        Parameters
        ----------
        analysis : af.Analysis
            An object that encapsulates the data and a log likelihood function.
        model : ModelMapper
            An object that represents possible instances of some model with a
            given dimensionality which is the number of free dimensions of the
            model.
        info : dict
            Optional dictionary containing information about the fit that can be loaded by the aggregator.

        Returns
        -------
        An object encapsulating how well the model fit the data, the best fit instance
        and an updated model with free parameters updated to represent beliefs
        produced by this fit.
        """

        self.paths.restore()

        if not os.path.exists(self.paths.has_completed_path):

            self.setup_log_file()
            self.save_model_info(model=model)
            self.save_parameter_names_file(model=model)
            self.save_metadata()
            self.save_info(info=info)
            self.save_search()
            self.save_model(model=model)

            self._fit(
                model=model,
                analysis=analysis,
            )
            open(self.paths.has_completed_path, "w+").close()

            samples = self.perform_update(model=model, analysis=analysis, during_analysis=False)

        else:

            samples = self.samples_from_model(model=model)
            self.paths.backup_zip_remove()

        return Result(samples=samples, previous_model=model)

    @abstractmethod
    def _fit(self, model, analysis):
        pass

    @property
    def tag(self):
        """Tag the output folder of the non-linear search, based on the non linear search settings"""
        raise NotImplementedError

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        name = "{}/{}".format(self.paths.name, extension)

        if remove_phase_tag:
            tag = ""
        else:
            tag = self.paths.tag

        new_instance = self.__class__(
            paths=Paths(
                name=name,
                tag=tag,
                folders=self.paths.folders,
                path_prefix=self.paths.path_prefix,
                non_linear_name=self.paths.non_linear_name,
                remove_files=self.paths.remove_files,
            ),
        )

        return new_instance

    @property
    def config_type(self):
        raise NotImplementedError()

    def config(self, section, attribute_name, attribute_type=str):
        """
        Get a config field from this search's section in non_linear.ini by a key and value type.

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
        return self.config_type.config_for(
            self.__class__.__name__).get(
            section,
            attribute_name,
            attribute_type
        )

    def perform_update(self, model, analysis, during_analysis):
        """Perform an update of the non-linear search results, which occurs every *iterations_per_update* of the
        non-linear search. The update performs the following tasks:

        1) Visualize the maximum log likelihood model.
        2) Backup the samples.
        3) Output the model results to the model.reults file.

        These task are performed every n updates, set by the relevent *task_every_update* variable, for example
        *visualize_every_update* and *backup_every_update*.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the non-linear search maximizes.
        during_analysis : bool
            If the update is during a non-linear search, in which case tasks are only performed after a certain number
             of updates and only a subset of visualization may be performed.
        """

        if self.should_backup() or not during_analysis:
            self.paths.backup()

        samples = self.samples_from_model(model=model)

        samples.write_table(filename=f"{self.paths.sym_path}/samples.csv")

        try:
            instance = samples.max_log_likelihood_instance
        except exc.FitException:
            return samples

        if self.should_visualize() or not during_analysis:
            analysis.visualize(instance=instance, during_analysis=during_analysis)

        if self.should_output_model_results() or not during_analysis:

            samples_text.results_to_file(
                samples=samples,
                file_results=self.paths.file_results,
                during_analysis=during_analysis
            )

        if not during_analysis:
            self.paths.backup_zip_remove()

        return samples

    def setup_log_file(self):

        if not len(self.log_file) == 0:
            log_path = "{}/{}".format(self.paths.output_path, self.log_file)
            logger.handlers = [logging.FileHandler(log_path)]
            logger.propagate = False
            # noinspection PyProtectedMember
            logger.level = logging._nameToLevel[
                conf.instance.general.get("output", "log_level", str)
                    .replace(" ", "")
                    .upper()
            ]

    def save_model_info(self, model):
        """Save the model.info file, which summarizes every parameter and prior."""
        with open(self.paths.file_model_info, "w+") as f:
            f.write(model.info)

    def save_parameter_names_file(self, model):
        """Create the param_names file listing every parameter's label and Latex tag, which is used for *GetDist*
        visualization.

        The parameter labels are determined using the label.ini and label_format.ini config files."""

        paramnames_names = model.parameter_names
        paramnames_labels = model_text.parameter_labels_from_model(model=model)

        parameter_name_and_label = []

        for i in range(model.prior_count):
            line = formatter.label_and_label_string(
                label0=paramnames_names[i], label1=paramnames_labels[i], whitespace=70
            )
            parameter_name_and_label += [f"{line}\n"]

        formatter.output_list_of_strings_to_file(
            file=self.paths.file_param_names, list_of_strings=parameter_name_and_label
        )

    def save_info(self, info):
        """
        Save the dataset associated with the phase
        """
        with open("{}/info.pickle".format(self.paths.pickle_path), "wb") as f:
            pickle.dump(info, f)

    def save_search(self):
        """
        Save the seawrch associated with the phase as a pickle
        """
        with open(self.paths.make_non_linear_pickle_path(), "w+b") as f:
            f.write(pickle.dumps(self))

    def save_model(self, model):
        """
        Save the model associated with the phase as a pickle
        """
        with open(self.paths.make_model_pickle_path(), "w+b") as f:
            f.write(pickle.dumps(model))

    def save_metadata(self):
        """
        Save metadata associated with the phase, such as the name of the pipeline, the
        name of the phase and the name of the dataset being fit
        """
        with open("{}/metadata".format(self.paths.make_path()), "a") as f:
            f.write(
                self.make_metadata_text()
            )

    @property
    def _default_metadata(self) -> Dict[str, str]:
        """
        A dictionary of metadata describing this phase, including the pipeline
        that it's embedded in.
        """
        return {
            "name": self.paths.name,
            "tag": self.paths.tag,
            "non_linear_search": type(self).__name__.lower(),
        }

    def make_metadata_text(self):
        return "\n".join(
            f"{key}={value or ''}"
            for key, value
            in {
                **self._default_metadata,
            }.items()
        )

    def initial_points_from_model(self, number_of_points, model):
        """Generate the initial points of the non-linear search, based on the initialize_method. The following methods
        can be used:

        ball (default):
            Walkers are initialized by randomly drawing unit values from a uniform distribution between the
            initialize_ball_lower_limit and initialize_ball_upper_limit values. It is recommended these limits are
            small, such that all walkers begin close to one another.
        prior:
            Walkers are initialized by randomly drawing unit values from a uniform distribution between 0 and 1,
            thus being distributed over the prior.

        Parameters
        ----------
        number_of_points : int
            The number of points in non-linear paramemter space which initial points are created for.
        model : ModelMapper
            An object that represents possible instances of some model with a given dimensionality which is the number
            of free dimensions of the model.
        """

        init_pos = np.zeros(shape=(number_of_points, model.prior_count))

        if self.initialize_method in "ball":

            for particle_index in range(number_of_points):

                init_pos[particle_index, :] = np.asarray(
                    model.random_vector_from_priors_within_limits(
                        lower_limit=self.initialize_ball_lower_limit,
                        upper_limit=self.initialize_ball_upper_limit
                    )
                )

        elif self.initialize_method in "prior":

            for particle_index in range(number_of_points):

                init_pos[particle_index, :] = np.asarray(
                    model.random_vector_from_priors
                )

        else:

            init_pos = None

        return init_pos

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

    def __eq__(self, other):
        return isinstance(other, NonLinearSearch) and self.__dict__ == other.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.paths.restore()


class Analysis:

    def __init__(self, log_likelihood_cap=None):
        self.log_likelihood_cap = log_likelihood_cap

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
