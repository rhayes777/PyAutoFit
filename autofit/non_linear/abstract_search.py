import copy
import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from os import path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from autoconf import conf
from autofit import exc
from autofit.mapper import model_mapper as mm
from autofit.non_linear import result as res
from autofit.non_linear import samples as samps
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.log import logger
from autofit.non_linear.parallel import SneakyPool
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.result import Result
from autofit.non_linear.timer import Timer


class NonLinearSearch(ABC):
    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag: Optional[str] = None,
            prior_passer: "PriorPasser" = None,
            initializer: Initializer = None,
            iterations_per_update: int = None,
            number_of_cores: int = 1,
            session: Optional[Session] = None,
            **kwargs
    ):
        """
        Abstract base class for non-linear searches.L

        This class sets up the file structure for the non-linear search, which are standardized across all non-linear
        searches.

        Parameters
        ------------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        prior_passer
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        session
            An SQLAlchemy session instance so the results of the model-fit are written to an SQLite database.
        """
        from autofit.non_linear.paths.database import DatabasePaths
        #
        name = name or ""
        path_prefix = path_prefix or ""

        self.path_prefix_no_unique_tag = path_prefix

        if unique_tag is not None:
            path_prefix = path.join(path_prefix, unique_tag)

        self.unique_tag = unique_tag

        if session is not None:
            paths = DatabasePaths(
                name=name,
                path_prefix=path_prefix,
                session=session,
                save_all_samples=kwargs.get(
                    "save_all_samples",
                    False
                ),
                unique_tag=unique_tag
            )
        else:
            paths = DirectoryPaths(
                name=name,
                path_prefix=path_prefix,
                unique_tag=unique_tag
            )

        self._paths = None
        self._timer = None

        self.paths: AbstractPaths = paths

        self.prior_passer = prior_passer or PriorPasser.from_config(
            config=self._config
        )

        self.force_pickle_overwrite = conf.instance["general"]["output"]["force_pickle_overwrite"]

        self.log_file = conf.instance["general"]["output"]["log_file"].replace(
            " ", ""
        )

        if initializer is None:
            self.initializer = Initializer.from_config(config=self._config)
        else:
            self.initializer = initializer

        self.iterations_per_update = iterations_per_update or self._config("updates", "iterations_per_update")

        if conf.instance["general"]["hpc"]["hpc_mode"]:
            self.iterations_per_update = conf.instance["general"]["hpc"]["iterations_per_update"]

        self.log_every_update = self._config("updates", "log_every_update")
        self.visualize_every_update = self._config(
            "updates", "visualize_every_update",
        )
        self.model_results_every_update = self._config(
            "updates", "model_results_every_update",
        )
        self.remove_state_files_at_end = self._config(
            "updates", "remove_state_files_at_end",
        )

        self.iterations = 0
        self.should_log = IntervalCounter(self.log_every_update)
        self.should_visualize = IntervalCounter(self.visualize_every_update)
        self.should_output_model_results = IntervalCounter(
            self.model_results_every_update
        )

        self.silence = self._config("printing", "silence")

        if conf.instance["general"]["hpc"]["hpc_mode"]:
            self.silence = True

        self.kwargs = kwargs

        for key, value in self.config_dict_search.items():
            setattr(self, key, value)

        try:
            for key, value in self.config_dict_run.items():
                setattr(self, key, value)
        except KeyError:
            pass

        self.number_of_cores = number_of_cores

    __identifier_fields__ = tuple()

    @property
    def timer(self):
        if self._timer is None:
            self._timer = Timer(
                self.paths.samples_path
            )
        return self._timer

    @property
    def paths(self) -> Optional[AbstractPaths]:
        return self._paths

    @paths.setter
    def paths(self, paths: Optional[AbstractPaths]):
        if paths is not None:
            paths.search = self
        self._paths = paths

    def copy_with_paths(
            self,
            paths
    ):
        search_instance = copy.copy(self)
        search_instance.paths = paths

        return search_instance

    class Fitness:
        def __init__(self, paths, model, analysis, samples_from_model, log_likelihood_cap=None):

            self.i = 0

            self.paths = paths
            self.analysis = analysis

            self.model = model
            self.samples_from_model = samples_from_model

            self.log_likelihood_cap = log_likelihood_cap

        def fit_instance(self, instance):

            log_likelihood = self.analysis.log_likelihood_function(instance=instance)

            if self.log_likelihood_cap is not None:
                if log_likelihood > self.log_likelihood_cap:
                    log_likelihood = self.log_likelihood_cap

            return log_likelihood

        def log_likelihood_from(self, parameter_list):

            instance = self.model.instance_from_vector(vector=parameter_list)
            log_likelihood = self.fit_instance(instance)

            return log_likelihood

        def log_posterior_from(self, parameter_list):

            log_likelihood = self.log_likelihood_from(parameter_list=parameter_list)
            log_prior_list = self.model.log_prior_list_from_vector(vector=parameter_list)

            return log_likelihood + sum(log_prior_list)

        def figure_of_merit_from(self, parameter_list):
            """The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. This varies
            between different `NonLinearSearch`s, for example:

                - The *Optimizer* *PySwarms* uses the chi-squared value, which is the -2.0*log_posterior.
                - The *MCMC* algorithm *Emcee* uses the log posterior.
                - Nested samplers such as *Dynesty* use the log likelihood.
            """
            raise NotImplementedError()

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
        def resample_figure_of_merit(self):
            """If a sample raises a FitException, this value is returned to signify that the point requires resampling or
             should be given a likelihood so low that it is discard."""
            return -np.inf

    def fit(
            self,
            model,
            analysis: "Analysis",
            info=None,
            pickle_files=None,
            log_likelihood_cap=None
    ) -> "Result":
        """
        Fit a model, M with some function f that takes instances of the
        class represented by model M and gives a score for their fitness.

        A model which represents possible instances with some dimensionality is fit.

        The analysis provides two functions. One visualises an instance of a model and the
        other scores an instance based on how well it fits some data. The search
        produces instances of the model by picking points in an N dimensional space.

        Parameters
        ----------
        log_likelihood_cap
        analysis : af.Analysis
            An object that encapsulates the data and a log likelihood function.
        model : ModelMapper
            An object that represents possible instances of some model with a
            given dimensionality which is the number of free dimensions of the
            model.
        info : dict
            Optional dictionary containing information about the fit that can be loaded by the aggregator.
        pickle_files : [str]
            Optional list of strings specifying the path and filename of .pickle files, that are copied to each
            model-fits pickles folder so they are accessible via the Aggregator.

        Returns
        -------
        An object encapsulating how well the model fit the data, the best fit instance
        and an updated model with free parameters updated to represent beliefs
        produced by this fit.
        """
        self.paths.model = model
        self.paths.unique_tag = self.unique_tag
        self.paths.restore()
        self.setup_log_file()

        if not self.paths.is_complete or self.force_pickle_overwrite:

            self.paths.save_all(
                search_config_dict=self.config_dict_search,
                info=info,
                pickle_files=pickle_files
            )
            analysis.save_attributes_for_aggregator(paths=self.paths)

        if not self.paths.is_complete:

            self.timer.start()

            self._fit(model=model, analysis=analysis, log_likelihood_cap=log_likelihood_cap)

            self.paths.completed()

            samples = self.perform_update(
                model=model, analysis=analysis, during_analysis=False
            )

            analysis.save_results_for_aggregator(paths=self.paths, model=model, samples=samples)
            self.paths.save_object("samples", samples)

        else:

            logger.info(f"{self.paths.name} already completed, skipping non-linear search.")
            samples = self.samples_from(model=model)

            if self.force_pickle_overwrite:

                self.paths.save_object("samples", samples)
                analysis.save_results_for_aggregator(paths=self.paths, model=model, samples=samples)

        self.paths.zip_remove()
        return analysis.make_result(samples=samples, model=model, search=self)

    @abstractmethod
    def _fit(self, model, analysis, log_likelihood_cap=None):
        pass

    @property
    def _class_config(self):
        return self.config_type[self.__class__.__name__]

    @property
    def config_dict_search(self):

        config_dict = copy.copy(self._class_config["search"]._dict)

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

        return config_dict

    @property
    def config_dict_run(self):

        config_dict = copy.copy(self._class_config["run"]._dict)

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

        return config_dict

    @property
    def config_dict_settings(self):
        return self._class_config["settings"]._dict

    @property
    def config_type(self):
        raise NotImplementedError()

    def _config(self, section, attribute_name):
        """
        Get a config field from this search's section in non_linear.ini by a key and value type.

        Parameters
        ----------
        attribute_name: str
            The analysis_path of the field

        Returns
        -------
        attribute
            An attribute for the key with the specified type.
        """
        return self._class_config[section][attribute_name]

    def perform_update(self, model, analysis, during_analysis):
        """
        Perform an update of the `NonLinearSearch` results, which occurs every *iterations_per_update* of the
        non-linear search. The update performs the following tasks:

        1) Visualize the maximum log likelihood model.
        2) Output the model results to the model.reults file.

        These task are performed every n updates, set by the relevent *task_every_update* variable, for example
        *visualize_every_update*

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.
        during_analysis : bool
            If the update is during a non-linear search, in which case tasks are only performed after a certain number
             of updates and only a subset of visualization may be performed.
        """

        self.iterations += self.iterations_per_update
        logger.info(
            f"{self.iterations} Iterations: Performing update (Visualization, outputting samples, etc.)."
        )

        self.timer.update()

        samples = self.samples_from(model=model)

        self.paths.save_samples(
            samples
        )

        if not during_analysis:
            self.plot_results(samples=samples)

        try:
            instance = samples.max_log_likelihood_instance
        except exc.FitException:
            return samples

        if self.should_visualize() or not during_analysis:

            analysis.visualize(
                paths=self.paths,
                instance=instance,
                during_analysis=during_analysis
            )

        if self.should_output_model_results() or not during_analysis:
            try:
                start = time.time()
                analysis.log_likelihood_function(instance=instance)
                log_likelihood_function_time = (time.time() - start)

                self.paths.save_summary(
                    samples=samples,
                    log_likelihood_function_time=log_likelihood_function_time
                )
            except exc.FitException:
                pass

        if not during_analysis and self.remove_state_files_at_end:
            try:
                self.remove_state_files()
            except FileNotFoundError:
                pass

        return samples

    def setup_log_file(self):

        if self.number_of_cores > 1:
            logger.disabled = True

        if conf.instance["general"]["output"]["log_to_file"]:

            if len(self.log_file) == 0:
                raise ValueError("In general.ini log_to_file is True, but log_file is an empty string. "
                                 "Either give log_file a name or set log_to_file to False.")

            log_path = path.join(self.paths.output_path, self.log_file)
            logger.handlers = [logging.FileHandler(log_path)]
            logger.propagate = False

    @property
    def samples_cls(self):
        raise NotImplementedError()

    def remove_state_files(self):
        pass

    def samples_from(self, model):
        raise NotImplementedError()

    def make_pool(self):
        """Make the pool instance used to parallelize a `NonLinearSearch` alongside a set of unique ids for every
        process in the pool. If the specified number of cores is 1, a pool instance is not made and None is returned.

        The pool cannot be set as an attribute of the class itself because this prevents pickling, thus it is generated
        via this function before calling the non-linear search.

        The pool instance is also set up with a list of unique pool ids, which are used during model-fitting to
        identify a 'master core' (the one whose id value is lowest) which handles model result output, visualization,
        etc."""

        if self.number_of_cores == 1:

            return None

        else:
            return mp.Pool(
                processes=self.number_of_cores
            )

    def make_sneaky_pool(
            self,
            fitness_function: Fitness
    ) -> Optional[SneakyPool]:
        """
        Create a pool for multiprocessing that uses slight-of-hand
        to avoid copying the fitness function between processes
        multiple times.

        Parameters
        ----------
        fitness_function
            An instance of a fitness class used to evaluate the
            likelihood that a particular model is correct

        Returns
        -------
        An implementation of a multiprocessing pool
        """
        if self.number_of_cores == 1:
            return None

        return SneakyPool(
            processes=self.number_of_cores,
            fitness=fitness_function
        )

    def __eq__(self, other):
        return isinstance(other, NonLinearSearch) and self.__dict__ == other.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def plot_results(self, samples):
        pass


class Analysis(ABC):

    def log_likelihood_function(self, instance):
        raise NotImplementedError()

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
        pass

    def save_attributes_for_aggregator(self, paths: AbstractPaths):
        pass

    def save_results_for_aggregator(self, paths: AbstractPaths, model: mm.CollectionPriorModel,
                                    samples: samps.OptimizerSamples):
        pass

    def make_result(self, samples, model, search):
        return res.Result(samples=samples, model=model, search=search)


class IntervalCounter:
    def __init__(self, interval):
        self.count = 0
        self.interval = interval

    def __call__(self):
        if self.interval == -1:
            return False
        self.count += 1
        return self.count % self.interval == 0


class PriorPasser:

    def __init__(self, sigma, use_errors, use_widths):
        """Class to package the API for prior passing.

        This class contains the parameters that controls how priors are passed from the results of one non-linear
        search to the next.

        Using the Phase API, we can pass priors from the result of one search to another follows:

            model_component.parameter = search1_result.model.model_component.parameter

        By invoking the 'model' attribute, the prior is passed following 3 rules:

            1) The new parameter uses a GaussianPrior. A GaussianPrior is ideal, as the 1D pdf results we compute at
               the end of a search are easily summarized as a Gaussian.

            2) The mean of the GaussianPrior is the median PDF value of the parameter estimated in search 1.

              This ensures that the initial sampling of the new search's non-linear starts by searching the region of
              non-linear parameter space that correspond to highest log likelihood solutions in the previous search.
              Thus, we're setting our priors to look in the 'correct' regions of parameter space.

            3) The sigma of the Gaussian will use the maximum of two values:

                    (i) the 1D error of the parameter computed at an input sigma value (default sigma=3.0).
                    (ii) The value specified for the profile in the 'config/priors/*.json' config
                         file's 'width_modifer' field (check these files out now).

               The idea here is simple. We want a value of sigma that gives a GaussianPrior wide enough to search a
               broad region of parameter space, so that the model can change if a better solution is nearby. However,
               we want it to be narrow enough that we don't search too much of parameter space, as this will be slow or
               risk leading us into an incorrect solution! A natural choice is the errors of the parameter from the
               previous search.

               Unfortunately, this doesn't always work. Modeling can be prone to an effect called 'over-fitting' where
               we underestimate the parameter errors. This is especially true when we take the shortcuts in early
               searchs - fast `NonLinearSearch` settings, simplified models, etc.

               Therefore, the 'width_modifier' in the json config files are our fallback. If the error on a parameter
               is suspiciously small, we instead use the value specified in the widths file. These values are chosen
               based on our experience as being a good balance broadly sampling parameter space but not being so narrow
               important solutions are missed.

        There are two ways a value is specified using the priors/width file:

            1) Absolute: In this case, the error assumed on the parameter is the value given in the config file. For
               example, if for the width on the parameter of a model component the width modifier reads "Absolute" with
               a value 0.05. This means if the error on the parameter was less than 0.05 in the previous search, the
               sigma of its GaussianPrior in this search will be 0.05.

            2) Relative: In this case, the error assumed on the parameter is the % of the value of the estimate value
               given in the config file. For example, if the parameter estimated in the previous search was 2.0, and the
               relative error in the config file reads "Relative" with a value 0.5, then the sigma of the GaussianPrior
               will be 50% of this value, i.e. sigma = 0.5 * 2.0 = 1.0.

        The PriorPasser allows us to customize at what sigma the error values the model results are computed at to
        compute the passed sigma values and customizes whether the widths in the config file, these computed errors,
        or both, are used to set the sigma values of the passed priors.

        The default values of the PriorPasser are found in the config file of every non-linear search, in the
        [prior_passer] section. All non-linear searches by default use a sigma value of 3.0, use_width=True and
        use_errors=True. We anticipate you should not need to change these values to get lens modeling to work
        proficiently!

        Example:

        Lets say in search 1 we fit a model, and we estimate that a parameter is equal to 4.0 +- 2.0, where the error
        value of 2.0 was computed at 3.0 sigma confidence. To pass this as a prior to search 2, we would write:

            model_component.parameter = result_1.model.model_component.parameter

        The prior on the parameter in search 2 would thus be a GaussianPrior, with mean=4.0 and
        sigma=2.0. If we had used a sigma value of 1.0 to compute the error, which reduced the estimate from 4.0 +- 2.0
        to 4.0 +- 0.5, the sigma of the Gaussian prior would instead be 0.5.

        If the error on the parameter in search 1 had been really small, lets say, 0.01, we would instead use the value
        of the parameter width in the priors config file to set sigma instead. Lets imagine the prior config file
        specifies that we use an "Absolute" value of 0.8 to link this prior. Then, the GaussianPrior in search 2 would
        have a mean=4.0 and sigma=0.8.

        If the prior config file had specified that we use an relative value of 0.8, the GaussianPrior in search 2 would
        have a mean=4.0 and sigma=3.2.
        """

        self.sigma = sigma
        self.use_errors = use_errors
        self.use_widths = use_widths

    @classmethod
    def from_config(cls, config):
        """Load the PriorPasser from a non_linear config file."""
        sigma = config("prior_passer", "sigma")
        use_errors = config("prior_passer", "use_errors")
        use_widths = config("prior_passer", "use_widths")
        return PriorPasser(sigma=sigma, use_errors=use_errors, use_widths=use_widths)
