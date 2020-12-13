import copy
import logging
import multiprocessing as mp
from os import path
import os
import pickle
import shutil
from abc import ABC, abstractmethod
from time import sleep
from typing import Dict

import numpy as np

from autoconf import conf
from autofit import exc
from autofit.mapper import model_mapper as mm
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.log import logger
from autofit.non_linear.paths import Paths, convert_paths
from autofit.non_linear import samples as samps
from autofit.non_linear.timer import Timer
from autofit.text import formatter
from autofit.text import text_util


class NonLinearSearch(ABC):
    @convert_paths
    def __init__(
            self,
            paths=None,
            prior_passer=None,
            initializer=None,
            iterations_per_update=None,
            number_of_cores=1,
    ):
        """Abstract base class for non-linear searches.

        This class sets up the file structure for the non-linear search, which are standardized across all non-linear
        searches.

        Parameters
        ------------
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        initializer : non_linear.initializer.Initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        """

        if paths.non_linear_name == "":
            paths.non_linear_name = self._config("tag", "name")

        if paths.non_linear_tag == "":
            paths.non_linear_tag_function = lambda: self.tag

        self.paths = paths
        if prior_passer is None:
            self.prior_passer = PriorPasser.from_config(config=self._config)
        else:
            self.prior_passer = prior_passer

        self.timer = Timer(paths=paths)

        self.force_pickle_overwrite = conf.instance["general"]["output"]["force_pickle_overwrite"]

        self.log_file = conf.instance["general"]["output"]["log_file"].replace(
            " ", ""
        )

        if initializer is None:
            self.initializer = Initializer.from_config(config=self._config)
        else:
            self.initializer = initializer

        self.iterations_per_update = (
            self._config("updates", "iterations_per_update")
            if iterations_per_update is None
            else iterations_per_update
        )

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

        self.number_of_cores = number_of_cores

        self._in_phase = False

    def copy_with_paths(
            self,
            paths
    ):
        search_instance = copy.copy(self)
        search_instance.paths = paths

        return search_instance

    class Fitness:
        def __init__(self, paths, model, analysis, samples_from_model, log_likelihood_cap=None, pool_ids=None):

            self.paths = paths
            self.max_log_likelihood = -np.inf
            self.analysis = analysis

            self.model = model
            self.samples_from_model = samples_from_model

            self.log_likelihood_cap = log_likelihood_cap
            self.pool_ids = pool_ids

        def fit_instance(self, instance):

            log_likelihood = self.analysis.log_likelihood_function(instance=instance)

            if self.log_likelihood_cap is not None:
                if log_likelihood > self.log_likelihood_cap:
                    log_likelihood = self.log_likelihood_cap

            if log_likelihood > self.max_log_likelihood:

                if self.pool_ids is not None:
                    if mp.current_process().pid != min(self.pool_ids):
                        return log_likelihood

                self.max_log_likelihood = log_likelihood

            return log_likelihood

        def log_likelihood_from_parameters(self, parameters):
            instance = self.model.instance_from_vector(vector=parameters)
            log_likelihood = self.fit_instance(instance)
            return log_likelihood

        def log_posterior_from_parameters(self, parameters):
            log_likelihood = self.log_likelihood_from_parameters(parameters=parameters)
            log_priors = self.model.log_priors_from_vector(vector=parameters)
            return log_likelihood + sum(log_priors)

        def figure_of_merit_from_parameters(self, parameters):
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

    def fit(self, model, analysis: "Analysis", info=None, pickle_files=None, log_likelihood_cap=None) -> "Result":
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
        pickle_files : [str]
            Optional list of strings specifying the path and filename of .pickle files, that are copied to each
            model-fits pickles folder so they are accessible via the Aggregator.

        Returns
        -------
        An object encapsulating how well the model fit the data, the best fit instance
        and an updated model with free parameters updated to represent beliefs
        produced by this fit.
        """

        try:
            os.makedirs(self.paths.samples_path)
        except FileExistsError:
            pass

        self.paths.restore()
        self.setup_log_file()

        if (not path.exists(self.paths.has_completed_path)) or \
                self.force_pickle_overwrite:

            self.save_model_info(model=model)
            self.save_parameter_names_file(model=model)
            self.save_metadata()
            self.save_info(info=info)
            self.save_search()
            self.save_model(model=model)
            self.move_pickle_files(pickle_files=pickle_files)
            analysis.save_attributes_for_aggregator(paths=self.paths)

        if not path.exists(self.paths.has_completed_path):

            # TODO : Better way to handle?
            self.timer.paths = self.paths
            self.timer.start()

            self._fit(model=model, analysis=analysis, log_likelihood_cap=log_likelihood_cap)
            open(self.paths.has_completed_path, "w+").close()

            samples = self.perform_update(
                model=model, analysis=analysis, during_analysis=False
            )

            analysis.save_results_for_aggregator(paths=self.paths, samples=samples)

        else:

            logger.info(f"{self.paths.name} already completed, skipping non-linear search.")
            samples = self.samples_via_csv_json_from_model(model=model)
            samples.write_table(filename=self.paths.samples_file)
            samples.info_to_json(filename=self.paths.info_file)
            self.save_samples(samples=samples)

            if self.force_pickle_overwrite:
                analysis.save_results_for_aggregator(paths=self.paths, samples=samples)

        self.paths.zip_remove()
        return Result(samples=samples, previous_model=model, search=self)

    @abstractmethod
    def _fit(self, model, analysis, log_likelihood_cap=None):
        pass

    @property
    def tag(self):
        """Tag the output folder of the non-linear search, based on the non linear search settings"""
        raise NotImplementedError

    def copy_with_name_extension(self, extension, path_prefix=None, remove_phase_tag=False):
        name = path.join(self.paths.name, extension)

        if path_prefix is None:
            path_prefix = self.paths.path_prefix

        if remove_phase_tag:
            tag = ""
        else:
            tag = self.paths.tag

        new_instance = self.__class__(
            paths=Paths(
                name=name,
                tag=tag,
                path_prefix=path_prefix,
                non_linear_name=self.paths.non_linear_name,
                remove_files=self.paths.remove_files,
            )
        )

        return new_instance

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
        return self.config_type[self.__class__.__name__][section][attribute_name]

    def perform_update(self, model, analysis, during_analysis):
        """Perform an update of the `NonLinearSearch` results, which occurs every *iterations_per_update* of the
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
        logger.info(f"{self.iterations} Iterations: Performing update (Visualization, outputting samples, etc.).")

        self.timer.update()

        samples = self.samples_via_sampler_from_model(model=model)
        samples.write_table(filename=self.paths.samples_file)
        samples.info_to_json(filename=self.paths.info_file)

        self.save_samples(samples=samples)

        try:
            instance = samples.max_log_likelihood_instance
        except exc.FitException:
            return samples

        if self.should_visualize() or not during_analysis:
            analysis.visualize(paths=self.paths, instance=instance, during_analysis=during_analysis)

        if self.should_output_model_results() or not during_analysis:

            text_util.results_to_file(
                samples=samples,
                filename=self.paths.file_results,
                during_analysis=during_analysis,
            )

            text_util.search_summary_to_file(samples=samples, filename=self.paths.file_search_summary)

        if not during_analysis and self.remove_state_files_at_end:
            try:
                self.remove_state_files()
            except FileNotFoundError:
                pass

        return samples

    def setup_log_file(self):

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

    def save_model_info(self, model):
        """Save the model.info file, which summarizes every parameter and prior."""
        with open(self.paths.file_model_info, "w+") as f:
            f.write(model.info)

    def save_parameter_names_file(self, model):
        """Create the param_names file listing every parameter's label and Latex tag, which is used for *corner.py*
        visualization.

        The parameter labels are determined using the label.ini and label_format.ini config files."""

        parameter_names = model.model_component_and_parameter_names
        parameter_labels = model.parameter_labels
        subscripts = model.subscripts
        parameter_labels_with_subscript = [f"{label}_{subscript}" for label, subscript in
                                           zip(parameter_labels, subscripts)]

        parameter_name_and_label = []

        for i in range(model.prior_count):
            line = formatter.add_whitespace(
                str0=parameter_names[i], str1=parameter_labels_with_subscript[i], whitespace=70
            )
            parameter_name_and_label += [f"{line}\n"]

        formatter.output_list_of_strings_to_file(
            file=self.paths.file_param_names, list_of_strings=parameter_name_and_label
        )

    def save_info(self, info):
        """
        Save the dataset associated with the phase
        """
        with open(path.join(self.paths.pickle_path, "info.pickle"), "wb") as f:
            pickle.dump(info, f)

    def save_search(self):
        """
        Save the seawrch associated with the phase as a pickle
        """
        with open(self.paths.make_search_pickle_path(), "w+b") as f:
            f.write(pickle.dumps(self))

    def save_model(self, model):
        """
        Save the model associated with the phase as a pickle
        """
        with open(self.paths.make_model_pickle_path(), "w+b") as f:
            f.write(pickle.dumps(model))

    def save_samples(self, samples):
        """
        Save the final-result samples associated with the phase as a pickle
        """
        with open(self.paths.make_samples_pickle_path(), "w+b") as f:
            f.write(pickle.dumps(samples))

    def save_metadata(self):
        """
        Save metadata associated with the phase, such as the name of the pipeline, the
        name of the phase and the name of the dataset being fit
        """
        with open(path.join(self.paths.make_path(), "metadata"), "a") as f:
            f.write(self.make_metadata_text())

    def move_pickle_files(self, pickle_files):
        """
        Move extra files a user has input the full path + filename of from the location specified to the
        pickles folder of the Aggregator, so that they can be accessed via the aggregator.
        """
        if pickle_files is not None:
            [shutil.copy(file, self.paths.pickle_path) for file in pickle_files]

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
            f"{key}={value or ''}" for key, value in {**self._default_metadata}.items()
        )

    def remove_state_files(self):
        pass

    def samples_via_sampler_from_model(self, model):
        raise NotImplementedError()

    def samples_via_csv_json_from_model(self, model):
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

            return None, None

        else:

            manager = mp.Manager()
            idQueue = manager.Queue()

            [idQueue.put(i) for i in range(self.number_of_cores)]

            pool = mp.Pool(
                processes=self.number_of_cores, initializer=init, initargs=(idQueue,)
            )
            ids = pool.map(f, range(self.number_of_cores))

            return pool, [id[1] for id in ids]

    def __eq__(self, other):
        return isinstance(other, NonLinearSearch) and self.__dict__ == other.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.paths.restore()


class Analysis(ABC):

    def log_likelihood_function(self, instance):
        raise NotImplementedError()

    def visualize(self, paths : Paths, instance, during_analysis):
        pass

    def save_attributes_for_aggregator(self, paths: Paths):
        pass

    def save_results_for_aggregator(self, paths: Paths, samples : samps.OptimizerSamples):
        pass

class Result:
    """
    @DynamicAttrs
    """

    def __init__(self, samples, previous_model, search=None):
        """
        The result of an optimization.

        Parameters
        ----------
        previous_model
            The model mapper from the stage that produced this result
        """

        self.samples = samples
        self.previous_model = previous_model
        self.search = search

        self.__model = None

        self._instance = (
            samples.max_log_likelihood_instance if samples is not None else None
        )

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
            tuples = self.samples.gaussian_priors_at_sigma(
                sigma=self.search.prior_passer.sigma
            )
            self.__model = self.previous_model.mapper_from_gaussian_tuples(
                tuples,
                use_errors=self.search.prior_passer.use_errors,
                use_widths=self.search.prior_passer.use_widths
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
            self.samples.gaussian_priors_at_sigma(sigma=self.search.prior_passer.sigma), a=a
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
            self.samples.gaussian_priors_at_sigma(sigma=self.search.prior_passer.sigma), r=r
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


class PriorPasser:

    def __init__(self, sigma, use_errors, use_widths):
        """Class to package the API for prior passing.

        This class contains the parameters that controls how priors are passed from the results of one non-linear
        search to the next.

        Using the Phase API, we can pass priors from the result of one phase to another follows:

            model_component.parameter = phase1_result.model.model_component.parameter

        By invoking the 'model' attribute, the prior is passed following 3 rules:

            1) The new parameter uses a GaussianPrior. A GaussianPrior is ideal, as the 1D pdf results we compute at
               the end of a phase are easily summarized as a Gaussian.

            2) The mean of the GaussianPrior is the median PDF value of the parameter estimated in phase 1.

              This ensures that the initial sampling of the new phase's non-linear starts by searching the region of
              non-linear parameter space that correspond to highest log likelihood solutions in the previous phase.
              Thus, we're setting our priors to look in the 'correct' regions of parameter space.

            3) The sigma of the Gaussian will use the maximum of two values:

                    (i) the 1D error of the parameter computed at an input sigma value (default sigma=3.0).
                    (ii) The value specified for the profile in the 'config/priors/*.json' config
                         file's 'width_modifer' field (check these files out now).

               The idea here is simple. We want a value of sigma that gives a GaussianPrior wide enough to search a
               broad region of parameter space, so that the model can change if a better solution is nearby. However,
               we want it to be narrow enough that we don't search too much of parameter space, as this will be slow or
               risk leading us into an incorrect solution! A natural choice is the errors of the parameter from the
               previous phase.

               Unfortunately, this doesn't always work. Modeling can be prone to an effect called 'over-fitting' where
               we underestimate the parameter errors. This is especially true when we take the shortcuts in early
               phases - fast `NonLinearSearch` settings, simplified models, etc.

               Therefore, the 'width_modifier' in the json config files are our fallback. If the error on a parameter
               is suspiciously small, we instead use the value specified in the widths file. These values are chosen
               based on our experience as being a good balance broadly sampling parameter space but not being so narrow
               important solutions are missed.

        There are two ways a value is specified using the priors/width file:

            1) Absolute: In this case, the error assumed on the parameter is the value given in the config file. For
               example, if for the width on the parameter of a model component the width modifier reads "Absolute" with
               a value 0.05. This means if the error on the parameter was less than 0.05 in the previous phase, the
               sigma of its GaussianPrior in this phase will be 0.05.

            2) Relative: In this case, the error assumed on the parameter is the % of the value of the estimate value
               given in the config file. For example, if the parameter estimated in the previous phase was 2.0, and the
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

        Lets say in phase 1 we fit a model, and we estimate that a parameter is equal to 4.0 +- 2.0, where the error
        value of 2.0 was computed at 3.0 sigma confidence. To pass this as a prior to phase 2, we would write:

            model_component.parameter = phase1.result.model.model_component.parameter

        The prior on the parameter in phase 2 would thus be a GaussianPrior, with mean=4.0 and
        sigma=2.0. If we had used a sigma value of 1.0 to compute the error, which reduced the estimate from 4.0 +- 2.0
        to 4.0 +- 0.5, the sigma of the Gaussian prior would instead be 0.5.

        If the error on the parameter in phase 1 had been really small, lets say, 0.01, we would instead use the value
        of the parameter width in the priors config file to set sigma instead. Lets imagine the prior config file
        specifies that we use an "Absolute" value of 0.8 to link this prior. Then, the GaussianPrior in phase 2 would
        have a mean=4.0 and sigma=0.8.

        If the prior config file had specified that we use an relative value of 0.8, the GaussianPrior in phase 2 would
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


def init(queue):
    global idx
    idx = queue.get()


def f(x):
    global idx
    process = mp.current_process()
    sleep(1)
    return idx, process.pid, x * x
