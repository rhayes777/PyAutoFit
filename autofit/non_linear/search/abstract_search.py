import copy
import logging
import multiprocessing as mp
import os
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

from autoconf import conf, cached_property
from autoconf.dictable import to_dict
from autofit import exc, jax_wrapper
from autofit.database.sqlalchemy_ import sa
from autofit.graphical import (
    MeanField,
    AnalysisFactor,
    _HierarchicalFactor,
    FactorApproximation,
)
from autofit.graphical.utils import Status
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import Collection
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.parallel import SneakyPool, SneakierPool
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.database import DatabasePaths
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.paths.sub_directory_paths import SubDirectoryPaths
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.result import Result
from autofit.non_linear.timer import Timer
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.analysis.combined import CombinedResult
from autofit.non_linear.analysis.indexed import IndexCollectionAnalysis
from autofit.non_linear.paths.null import NullPaths
from autofit.graphical.declarative.abstract import PriorFactor
from autofit.graphical.expectation_propagation import AbstractFactorOptimiser

from autofit.non_linear.fitness import get_timeout_seconds

logger = logging.getLogger(__name__)


def check_cores(func):
    """
    Checks how many cores the search has been configured to
    use and then returns None instead of calling the pool
    creation function in the case that only one core has
    been set.

    Parameters
    ----------
    func
        A function that creates a pool

    Returns
    -------
    None or a pool
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.number_of_cores == 1:
            return None
        return func(self, *args, **kwargs)

    return wrapper


class NonLinearSearch(AbstractFactorOptimiser, ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        initializer: Initializer = None,
        iterations_per_update: int = None,
        number_of_cores: int = 1,
        session: Optional[sa.orm.Session] = None,
        paths: Optional[AbstractPaths] = None,
        **kwargs,
    ):
        """
        Abstract base class for non-linear searches.L

        This class sets up the file structure for the non-linear search, which are standardized across all non-linear
        searches.

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        session
            An SQLAlchemy session instance so the results of the model-fit are written to an SQLite database.
        """
        super().__init__()

        from autofit.non_linear.paths.database import DatabasePaths

        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            self.is_master = comm.Get_rank() == 0

            logger.debug(f"Creating non-linear search: {comm.Get_rank()}")

        except ModuleNotFoundError:
            self.is_master = True
            logger.debug(f"Creating non-linear search")

        if name:
            path_prefix = Path(path_prefix or "")

        self.path_prefix = path_prefix

        self.path_prefix_no_unique_tag = path_prefix

        self._logger = None

        if unique_tag is not None and path_prefix is not None:
            path_prefix = path_prefix / unique_tag

        self.unique_tag = unique_tag

        if paths:
            self.paths = paths
        elif session is not None:
            logger.debug("Session found. Using database.")
            self.paths = DatabasePaths(
                name=name,
                path_prefix=path_prefix,
                session=session,
                save_all_samples=kwargs.get("save_all_samples", False),
                unique_tag=unique_tag,
            )
        elif name is not None or path_prefix:
            logger.debug("Session not found. Using directory output.")
            self.paths = DirectoryPaths(
                name=name, path_prefix=path_prefix, unique_tag=unique_tag
            )
        else:
            self.paths = NullPaths()

        self.force_pickle_overwrite = conf.instance["general"]["output"][
            "force_pickle_overwrite"
        ]
        self.skip_save_samples = kwargs.get("skip_save_samples")
        if self.skip_save_samples is None:
            self.skip_save_samples = conf.instance["general"]["output"].get(
                "skip_save_samples"
            )

        self.force_visualize_overwrite = conf.instance["general"]["output"][
            "force_visualize_overwrite"
        ]

        if initializer is None:
            self.logger.debug("Creating initializer ")
            self.initializer = Initializer.from_config(config=self._config)
        else:
            self.initializer = initializer

        self.iterations_per_update = iterations_per_update or self._config(
            "updates", "iterations_per_update"
        )

        if conf.instance["general"]["hpc"]["hpc_mode"]:
            self.iterations_per_update = conf.instance["general"]["hpc"][
                "iterations_per_update"
            ]

        self.remove_state_files_at_end = self._config(
            "updates",
            "remove_state_files_at_end",
        )

        self.iterations = 0

        self.should_profile = conf.instance["general"]["profiling"]["should_profile"]

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

        if jax_wrapper.use_jax:
            self.number_of_cores = 1
            logger.warning(f"JAX is enabled. Setting number of cores to 1.")

        self.number_of_cores = number_of_cores

        if number_of_cores > 1 and any(
            os.environ.get(key) != "1"
            for key in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ):
            warnings.warn(
                exc.SearchWarning(
                    """
                    The non-linear search is using multiprocessing (number_of_cores>1). 

                    However, the following environment variables have not been set to 1:

                    OPENBLAS_NUM_THREADS
                    MKL_NUM_THREADS
                    OMP_NUM_THREADS
                    VECLIB_MAXIMUM_THREADS
                    NUMEXPR_NUM_THREADS

                    This can lead to performance issues, because both the non-linear search and libraries that may be
                    used in your `log_likelihood_function` evaluation (e.g. NumPy, SciPy, scikit-learn) may attempt to
                    parallelize over all cores available.

                    This will lead to slow-down, due to overallocation of tasks over the CPUs.

                    To mitigate this, set the environment variables to 1 via the following command on your
                    bash terminal / command line:

                    export OPENBLAS_NUM_THREADS=1
                    export MKL_NUM_THREADS=1
                    export OMP_NUM_THREADS=1
                    export VECLIB_MAXIMUM_THREADS=1
                    export NUMEXPR_NUM_THREADS=1

                    This means only the non-linear search is parallelized over multiple cores.

                    If you "know what you are doing" and do not want these environment variables to be set to one, you 
                    can disable this warning by changing the following entry in the config files:

                    `config -> general.yaml -> parallel: -> warn_environment_variable=False`
                    """
                )
            )

        self.optimisation_counter = Counter()

    __identifier_fields__ = tuple()

    def optimise(
        self,
        factor_approx: FactorApproximation,
        status: Status = Status(),
    ) -> Tuple[MeanField, Status]:
        """
        Perform optimisation for expectation propagation. Currently only
        applicable for ModelFactors created by the declarative interface.

        1. Analysis and model classes are extracted from the factor.
        2. Priors are updated from the mean field.
        3. Analysis and model are fit as usual.
        4. A new mean field is constructed with the (posterior) 'linking' priors.
        5. Projection is performed to produce an updated EPMeanField object.

        Output directories are generated according to the factor and the number
        of the search. For example a factor called "factor" would output:

        factor/optimization_0/<identifier>
        factor/optimization_1/<identifier>
        factor/optimization_2/<identifier>

        For the first, second and third optimizations respectively.

        Parameters
        ----------
        factor_approx
            A collection of messages defining the current best approximation to
            some global model
        status

        Returns
        -------
        An updated approximation to the model having performed optimisation on
        a single factor.
        """

        factor = factor_approx.factor

        _ = status
        if not isinstance(factor, (AnalysisFactor, PriorFactor, _HierarchicalFactor)):
            raise NotImplementedError(
                f"Optimizer {self.__class__.__name__} can only be applied to"
                f" AnalysisFactors, HierarchicalFactors and PriorFactors"
            )

        model = factor.prior_model.mapper_from_prior_arguments(
            {
                prior: prior.with_message(message)
                for prior, message in factor_approx.cavity_dist.arguments.items()
            }
        )

        analysis = factor.analysis

        number = self.optimisation_counter[factor.name]

        self.optimisation_counter[factor.name] += 1

        self.paths = SubDirectoryPaths(
            parent=self.paths,
            analysis_name=f"{factor.name}/optimization_{number}",
            is_flat=True,
        )

        result = self.fit(model=model, analysis=analysis)

        new_model_dist = MeanField.from_priors(result.projected_model.priors)

        status.result = result

        return new_model_dist, status

    @property
    def name(self):
        return self.paths.name

    def __getstate__(self):
        """
        Remove the logger for pickling
        """
        state = self.__dict__.copy()
        if "_logger" in state:
            del state["_logger"]
        if "paths" in state:
            del state["paths"]
        return state

    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = None
        if self._logger is None:
            logger_ = logging.getLogger(self.name)
            self._logger = logger_
        return self._logger

    @property
    def timer(self) -> Optional[Timer]:
        """
        Returns the timer of the search, which is used to output informaiton such as how long the search took and
        how much parallelization sped up the search time.

        If the search is running in `NullPaths` mode, meaning that no output is written to the hard-disk, the timer
        is disabled and a `None` is returned.

        Returns
        -------
        An object which times the non-linear search.
        """
        try:
            return Timer(self.paths.search_internal_path)
        except TypeError:
            pass

    @property
    def paths(self) -> Optional[AbstractPaths]:
        return self._paths

    @paths.setter
    def paths(self, paths: Optional[AbstractPaths]):
        if paths is not None:
            paths.search = self
        self._paths = paths

    def copy_with_paths(self, paths):
        self.logger.debug(f"Creating a copy of {self._paths.name}")
        search_instance = copy.copy(self)
        search_instance.paths = paths
        search_instance._logger = None

        return search_instance

    @property
    def using_mpi(self) -> bool:
        """
        Whether the search is being performing using MPI for parallelisation or not.

        This is performed by checking the size of the MPI.COMM_WORLD communicator.

        Returns
        -------
        A bool indicating if the search is using MPI.
        """

        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            return comm.size > 1

        except ModuleNotFoundError:
            return False

    def fit_sequential(
        self,
        model: AbstractPriorModel,
        analysis: IndexCollectionAnalysis,
        info: Optional[Dict] = None,
    ) -> CombinedResult:
        """
        Fit multiple analyses contained within the analysis sequentially.

        This can be useful for avoiding very high dimensional parameter spaces.

        Parameters
        ----------
        analysis
            Multiple analyses that are fit sequentially
        model
            An object that represents possible instances of some model with a
            given dimensionality which is the number of free dimensions of the
            model.
        info
            Optional dictionary containing information about the fit that can be loaded by the aggregator.

        Returns
        -------
        An object combining the results of each individual optimisation.

        Raises
        ------
        AssertionError
            If the model has 0 dimensions.
        ValueError
            If the analysis is not a combined analysis
        """
        results = []

        _paths = self.paths
        original_name = self.paths.name or "analysis"

        model = analysis.modify_model(model=model)

        try:
            if not isinstance(model, Collection):
                model = [model for _ in range(len(analysis.analyses))]
        except AttributeError:
            raise ValueError(
                f"Analysis with type {type(analysis)} is not supported by fit_sequential"
            )

        for i, (model, analysis) in enumerate(zip(model, analysis.analyses)):
            self.paths = copy.copy(_paths)
            self.paths.name = f"{original_name}/{i}"
            results.append(
                self.fit(
                    model=model,
                    analysis=analysis,
                    info=info,
                )
            )
        self.paths = _paths
        return CombinedResult(results)

    def fit(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        info: Optional[Dict] = None,
        bypass_nuclear_if_on: bool = False,
    ) -> Union["Result", List["Result"]]:
        """
        Fit a model, M with some function f that takes instances of the
        class represented by model M and gives a score for their fitness.

        A model which represents possible instances with some dimensionality is fit.

        The analysis provides two functions. One visualises an instance of a model and the
        other scores an instance based on how well it fits some data. The search
        produces instances of the model by picking points in an N dimensional space.

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.
        info
            Optional dictionary containing information about the fit that can be saved in the `files` folder
            (e.g. as `files/info.json`) and can be loaded via the database.
        bypass_nuclear_if_on
            If nuclear mode is on (environment variable "PYAUTOFIT_NUCLEAR_MODE=1") passing this as True will
            bypass it.

        Returns
        -------
        An object encapsulating how well the model fit the data, the best fit instance
        and an updated model with free parameters updated to represent beliefs
        produced by this fit.

        Raises
        ------
        AssertionError
            If the model has 0 dimensions.
        """

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            from autofit.non_linear.search.nest.nautilus.search import Nautilus

            if isinstance(self, Nautilus):
                from autofit.non_linear.search.nest.dynesty.search.static import (
                    DynestyStatic,
                )

                from pathlib import Path

                search = DynestyStatic(
                    paths=self.paths,
                    unique_tag=self.unique_tag,
                    number_of_cores=self.number_of_cores,
                )

                return search.fit(model=model, analysis=analysis, info=info)

        self.check_model(model=model)

        model = analysis.modify_model(model)
        self.paths.model = model
        self.paths.unique_tag = self.unique_tag

        if self.is_master:
            self.paths.restore()

        model.freeze()
        analysis = analysis.modify_before_fit(paths=self.paths, model=model)
        model.unfreeze()

        if self.is_master:
            self.pre_fit_output(
                analysis=analysis,
                model=model,
                info=info,
            )

        if not self.paths.is_complete:
            result = self.start_resume_fit(
                analysis=analysis,
                model=model,
            )
        else:
            result = self.result_via_completed_fit(
                analysis=analysis,
                model=model,
            )

        if self.is_master:
            analysis = analysis.modify_after_fit(
                paths=self.paths, model=model, result=result
            )

            self.post_fit_output(
                bypass_nuclear_if_on=bypass_nuclear_if_on,
            )

        return result

    def pre_fit_output(
        self, analysis: Analysis, model: AbstractPriorModel, info: Optional[Dict] = None
    ):
        """
        Outputs attributes of fit before the non-linear search begins.

        The following attributes of a fit may be output before the search begins:

        - The model composition, which is output as a .json file (`files/model.json`).

        - The non-linear search settings, which are output as a .json file (`files/search.json`).

        - Custom attributes of the analysis defined via the `save_attributes` method of the analysis class, for
        example the data (e.g. `files/data.json`).

        - Custom Visualization associated with the analysis, defined via the `visualize_before_fit`
        and `visualize_before_fit_combined` methods. This is typically quantities that do not change during the
        model-fit (e.g. the data).

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.
        info
            Optional dictionary containing information about the fit that can be saved in the `files` folder
            (e.g. as `files/info.json`) and can be loaded via the database.
        """

        self.logger.info(f"The output path of this fit is {self.paths.output_path}")

        if not self.paths.is_complete or self.force_pickle_overwrite:
            self.logger.info(
                f"Outputting pre-fit files (e.g. model.info, visualization)."
            )

            self.paths.save_all(
                search_config_dict=self.config_dict_search,
                info=info,
            )
            analysis.save_attributes(paths=self.paths)

        if analysis.should_visualize(paths=self.paths):
            analysis.visualize_before_fit(
                paths=self.paths,
                model=model,
            )
            analysis.visualize_before_fit_combined(
                analyses=None,
                paths=self.paths,
                model=model,
            )

            timeout_seconds = get_timeout_seconds()

            if timeout_seconds is not None:
                logger.info(
                    f"\n\n ***Log Likelihood Function timeout is "
                    f"turned on and set to {timeout_seconds} seconds.***\n"
                )

    def start_resume_fit(self, analysis: Analysis, model: AbstractPriorModel) -> Result:
        """
        Start a non-linear search from scratch, or resumes one which was previously terminated mid-way through.

        If the search is resumed, the model-fit will begin by loading the samples from the previous search and
        from where it left off.

        After the search is completed, a `.completed` file is output so that if the search is resumed in the future
        it is not repeated and results are loaded via the `update_completed_fit` method.

        Results are also output to hard-disk in the `files` folder via the `save_results` method of the analysis class.

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.

        Returns
        -------
        The result of the non-linear search, which includes the best-fit model instance and best-fit log likelihood
        and errors on the model parameters.
        """
        if self.is_master:
            if not isinstance(self.paths, DatabasePaths) and not isinstance(
                self.paths, NullPaths
            ):
                self.timer.start()

        model.freeze()
        search_internal = self._fit(
            model=model,
            analysis=analysis,
        )
        samples = self.perform_update(
            model=model,
            analysis=analysis,
            search_internal=search_internal,
            during_analysis=False,
        )

        result = analysis.make_result(
            samples=samples,
        )

        if self.is_master:
            analysis.save_results(paths=self.paths, result=result)

        model.unfreeze()

        self.paths.completed()

        return result

    def result_via_completed_fit(
        self, analysis: Analysis, model: AbstractPriorModel, search_internal=None
    ) -> Result:
        """
        Returns the result of the non-linear search of a completed model-fit.

        The result contains the non-linear search samples, which are loaded from the searches internal results,
        or the `samples.csv` file if the internal results are not available.

        Optional tasks can be performed to update the results of the model-fit on hard-disk depending on the following
        entries of the `general.yaml` config file's `output` section:

        ` `force_visualize_overwrite=True`: the visualization of the model-fit is performed again (e.g. to
        add new visualizations or replot figures with a different source code).

        - `force_pickle_overwrite=True`: the output files of the model-fit are recreated (e.g. to add a new attribute
        that was previously not output).

        Parameters
        ----------
        analysis
            An object that encapsulates the data and a log likelihood function which fits the model to the data
            via the non-linear search.
        model
            The model that is fitted to the data, which is used by the non-linear search to create instances of
            the model that are fitted to the data via the log likelihood function.
        Returns
        -------
        The result of the non-linear search, which includes the best-fit model instance and best-fit log likelihood
        and errors on the model parameters.
        """

        model.freeze()
        samples = self.samples_from(model=model)

        result = analysis.make_result(
            samples=samples,
        )

        if self.is_master:
            self.logger.info(f"Fit Already Completed: skipping non-linear search.")

            if self.force_visualize_overwrite:
                self.perform_visualization(
                    model=model,
                    analysis=analysis,
                    search_internal=search_internal,
                    during_analysis=False,
                )

            if self.force_pickle_overwrite:
                self.logger.info("Forcing pickle overwrite")

                if not self.skip_save_samples:
                    self.paths.save_json("samples_summary", to_dict(samples.summary()))

                analysis.save_results(paths=self.paths, result=result)

        model.unfreeze()

        return result

    def post_fit_output(self, bypass_nuclear_if_on: bool):
        """
        Cleans up the output folderds after a completed non-linear search.

        The main task this performs is removing the folder containing the results of a non-linear search such that only
        its corresponding `.zip` file is left. This is use for supercomputers, where users often have a file limit on
        the number of files they can store in their home directory, so storing them all in just a .zip file is
        advantageous.

        This only occurs if `remove_files=False` in the `general.yaml` config file's `output` section.

        Parameters
        ----------
        bypass_nuclear_if_on
            Whether to use nuclear mode to delete a lot of files (see nuclear mode description).
        """
        self.logger.info("Removing all files except for .zip file")
        self.paths.zip_remove()

        if not bypass_nuclear_if_on:
            self.paths.zip_remove_nuclear()

    @abstractmethod
    def _fit(self, model: AbstractPriorModel, analysis: Analysis):
        pass

    def check_model(self, model: AbstractPriorModel):
        if model is not None and model.prior_count == 0:
            raise AssertionError("Model has no priors! Cannot fit a 0 dimension model.")

    def config_dict_with_test_mode_settings_from(self, config_dict: Dict) -> Dict:
        return config_dict

    @property
    def _class_config(self) -> Dict:
        return self.config_type[self.__class__.__name__]

    @cached_property
    def config_dict_search(self) -> Dict:
        config_dict = copy.deepcopy(self._class_config["search"])

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

        return config_dict

    @cached_property
    def config_dict_run(self) -> Dict:
        config_dict = copy.deepcopy(self._class_config["run"])

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            logger.warning(f"TEST MODE ON: SEARCH WILL SKIP SAMPLING\n\n")

            config_dict = self.config_dict_with_test_mode_settings_from(
                config_dict=config_dict
            )

        return config_dict

    @property
    def config_dict_settings(self) -> Dict:
        return self._class_config["settings"]

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

    def perform_update(
        self,
        model: AbstractPriorModel,
        analysis: Analysis,
        during_analysis: bool,
        search_internal=None,
    ) -> Samples:
        """
        Perform an update of the non-linear search's model-fitting results.

        This occurs every `iterations_per_update` of the non-linear search and once it is complete.

        The update performs the following tasks (if the settings indicate they should be performed):

        1) Visualize the search results (e.g. a cornerplot).
        2) Visualize the maximum log likelihood model using model-specific visualization implented via the `Analysis`
           object.
        3) Perform profiling of the analysis object `log_likelihood_function` and ouptut run-time information.
        4) Output the `search.summary` file which contains information on model-fitting so far.
        5) Output the `model.results` file which contains a concise text summary of the model results so far.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.
        during_analysis
            If the update is during a non-linear search, in which case tasks are only performed after a certain number
            of updates and only a subset of visualization may be performed.
        """

        self.iterations += self.iterations_per_update
        if during_analysis:
            self.logger.info(
                f"Fit Still Running: Updating results after {self.iterations} iterations (see "
                f"output folder for latest visualization, samples, etc.)"
            )
        else:
            self.logger.info(
                f"Fit Complete: Updating final results (see "
                f"output folder for final visualization, samples, etc.)"
            )

        if not isinstance(self.paths, DatabasePaths) and not isinstance(
            self.paths, NullPaths
        ):
            self.timer.update()

        samples = self.samples_from(model=model, search_internal=search_internal)

        try:
            instance = samples.max_log_likelihood()
        except exc.FitException:
            return samples

        if self.is_master:
            self.paths.samples_to_csv(samples=samples)

            if not self.skip_save_samples:
                self.paths.save_json("samples_summary", to_dict(samples.summary()))

            self.perform_visualization(
                model=model,
                analysis=analysis,
                during_analysis=during_analysis,
                search_internal=search_internal,
            )

            if self.should_profile:
                self.logger.debug("Profiling Maximum Likelihood Model")
                analysis.profile_log_likelihood_function(
                    paths=self.paths,
                    instance=instance,
                )

            self.logger.debug("Outputting model result")
            try:
                log_likelihood_function = jax_wrapper.jit(
                    analysis.log_likelihood_function,
                )
                log_likelihood_function(instance=instance)

                start = time.time()
                log_likelihood_function(instance=instance)
                log_likelihood_function_time = time.time() - start

                self.paths.save_summary(
                    samples=samples,
                    log_likelihood_function_time=log_likelihood_function_time,
                )
            except exc.FitException:
                pass

            if not during_analysis and self.remove_state_files_at_end:
                self.logger.debug("Removing state files")
                try:
                    self.remove_state_files()
                except FileNotFoundError:
                    pass

        return samples

    def perform_visualization(
        self,
        model: AbstractPriorModel,
        analysis: AbstractPriorModel,
        during_analysis: bool,
        search_internal=None,
    ):
        """
        Perform visualization of the non-linear search's model-fitting results.

        This occurs every `iterations_per_update` of the non-linear search, when the search is complete and can
        also be forced to occur even though a search is completed on a rerun, to update the visualization
        with different `matplotlib` settings.

        The update performs the following tasks (if the settings indicate they should be performed):

        1) Visualize the search results (e.g. a cornerplot).
        2) Visualize the maximum log likelihood model using model-specific visualization implented via the `Analysis`
           object.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.
        during_analysis
            If the update is during a non-linear search, in which case tasks are only performed after a certain number
            of updates and only a subset of visualization may be performed.
        """
        samples = self.samples_from(model=model, search_internal=search_internal)

        try:
            instance = samples.max_log_likelihood()
        except exc.FitException:
            return samples

        if analysis.should_visualize(paths=self.paths, during_analysis=during_analysis):
            if not isinstance(self.paths, NullPaths):
                self.plot_results(samples=samples)

        self.logger.debug("Visualizing")
        if analysis.should_visualize(paths=self.paths, during_analysis=during_analysis):
            analysis.visualize(
                paths=self.paths, instance=instance, during_analysis=during_analysis
            )
            analysis.visualize_combined(
                analyses=None,
                paths=self.paths,
                instance=instance,
                during_analysis=during_analysis,
            )

    @property
    def samples_cls(self):
        raise NotImplementedError()

    def remove_state_files(self):
        pass

    def samples_from(self, model: AbstractPriorModel, search_internal=None) -> Samples:
        """
        Loads the samples of a non-linear search from its output files.

        The samples can be loaded from one of two files, which are attempted to be loading in the following order:

        1) Load via the internal results of the non-linear search, which are specified to that search's outputs
           (e.g. the .hdf file output by the MCMC method `emcee`).

        2) Load via the `samples.csv` and `samples_info.json` files of the search, which are outputs that are the
           same for all non-linear searches as they are homogenized by autofit.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        """
        try:
            return self.samples_via_internal_from(
                model=model, search_internal=search_internal
            )
        except (FileNotFoundError, NotImplementedError, AttributeError):
            return self.samples_via_csv_from(model=model)

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ):
        raise NotImplementedError

    def samples_via_csv_from(self, model: AbstractPriorModel) -> Samples:
        """
        Returns a `Samples` object from the `samples.csv` and `samples_info.json` files.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The samples in csv format are already converted to the autofit format, where samples are lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        return self.samples_cls.from_csv(
            paths=self.paths,
            model=model,
        )

    @check_cores
    def make_pool(self):
        """Make the pool instance used to parallelize a `NonLinearSearch` alongside a set of unique ids for every
        process in the pool. If the specified number of cores is 1, a pool instance is not made and None is returned.

        The pool cannot be set as an attribute of the class itself because this prevents pickling, thus it is generated
        via this function before calling the non-linear search.

        The pool instance is also set up with a list of unique pool ids, which are used during model-fitting to
        identify a 'master core' (the one whose id value is lowest) which handles model result output, visualization,
        etc."""
        self.logger.info("...using pool")
        return mp.Pool(processes=self.number_of_cores)

    @check_cores
    def make_sneaky_pool(self, fitness: Fitness) -> Optional[SneakyPool]:
        """
        Create a pool for multiprocessing that uses slight-of-hand
        to avoid copying the fitness function between processes
        multiple times.

        Parameters
        ----------
        fitness
            An instance of a fitness class used to evaluate the
            likelihood that a particular model is correct

        Returns
        -------
        An implementation of a multiprocessing pool
        """
        self.logger.warning(
            "...using SneakyPool. This copies the likelihood function "
            "to each process on instantiation to avoid copying multiple "
            "times."
        )
        return SneakyPool(
            processes=self.number_of_cores, paths=self.paths, fitness=fitness
        )

    def make_sneakier_pool(self, fitness_function: Fitness, **kwargs) -> SneakierPool:
        if self.is_master:
            self.logger.info(f"number of cores == {self.number_of_cores}")

        if self.is_master:
            if self.number_of_cores > 1:
                self.logger.info("Creating SneakierPool...")
            else:
                self.logger.info("Creating multiprocessing Pool of size 1...")

        pool = SneakierPool(
            processes=self.number_of_cores, fitness=fitness_function, **kwargs
        )

        return pool

    def __eq__(self, other):
        return isinstance(other, NonLinearSearch) and self.__dict__ == other.__dict__

    def plot_results(self, samples):
        pass
