import numpy as np
import logging
import os
import sys
from typing import Optional

from autofit import jax_wrapper
from autofit.database.sqlalchemy_ import sa

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.search.nest import abstract_nest
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.nest import SamplesNest
from autofit.plot.output import Output

logger = logging.getLogger(__name__)


def prior_transform(cube, model):
    return model.vector_from_unit_vector(unit_vector=cube, ignore_prior_limits=True)


class Nautilus(abstract_nest.AbstractNest):
    __identifier_fields__ = (
        "n_live",
        "n_update",
        "enlarge_per_dim",
        "n_points_min",
        "split_threshold",
        "n_networks",
        "n_like_new_bound",
        "seed",
        "n_shell",
        "n_eff",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        iterations_per_update: int = None,
        number_of_cores: int = None,
        session: Optional[sa.orm.Session] = None,
        **kwargs
    ):
        """
        A Nautilus non-linear search.

        Nautilus is an optional requirement and must be installed manually via the command `pip install ultranest`.
        It is optional as it has certain dependencies which are generally straight forward to install (e.g. Cython).

        For a full description of Nautilus checkout its Github and documentation webpages:

        https://github.com/johannesulf/nautilus
        https://nautilus-sampler.readthedocs.io/en/stable/index.html

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        iterations_per_update
            The number of iterations performed between update (e.g. output latest model to hard-disk, visualization).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs,
        )

        self.logger.debug("Creating Nautilus Search")

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using the search and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the `NonLinearSearch` seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.

        Returns
        -------
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted ssamples of the fit.
        """

        fitness = Fitness(
            model=model,
            analysis=analysis,
            fom_is_log_likelihood=True,
            resample_figure_of_merit=-1.0e99,
        )

        if not isinstance(self.paths, NullPaths):
            checkpoint_exists = os.path.exists(self.checkpoint_file)
        else:
            checkpoint_exists = False

        if checkpoint_exists:
            self.logger.info(
                "Resuming Nautilus non-linear search (previous samples found)."
            )

        else:
            self.logger.info(
                "Starting new Nautilus non-linear search (no previous samples found)."
            )

        if (
            self.config_dict.get("force_x1_cpu")
            or self.kwargs.get("force_x1_cpu")
            or jax_wrapper.use_jax
        ):
            search_internal = self.fit_x1_cpu(
                fitness=fitness,
                model=model,
                analysis=analysis,
                checkpoint_exists=checkpoint_exists,
            )
        else:
            if not self.using_mpi:
                search_internal = self.fit_multiprocessing(
                    fitness=fitness,
                    model=model,
                    analysis=analysis,
                    checkpoint_exists=checkpoint_exists,
                )
            else:
                search_internal = self.fit_mpi(
                    fitness=fitness,
                    model=model,
                    analysis=analysis,
                    checkpoint_exists=checkpoint_exists,
                )

        if self.checkpoint_file is not None:
            os.remove(self.checkpoint_file)

        return search_internal

    @property
    def sampler_cls(self):
        try:
            from nautilus import Sampler

            return Sampler
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "\n--------------------\n"
                "You are attempting to perform a model-fit using Nautilus. \n\n"
                "However, the optional library Nautilus (https://nautilus-sampler.readthedocs.io/en/stable/index.html) is "
                "not installed.\n\n"
                "Install it via the command `pip install nautilus-sampler==0.7.2`.\n\n"
                "----------------------"
            )

    @property
    def checkpoint_file(self):
        """
        The path to the file used for checkpointing.

        If autofit is not outputting results to hard-disk (e.g. paths is `NullPaths`), this function is bypassed.
        """
        try:
            return self.paths.search_internal_path / "checkpoint.hdf5"
        except TypeError:
            pass

    def fit_x1_cpu(self, fitness, model, analysis, checkpoint_exists: bool):
        """
        Perform the non-linear search, using one CPU core.

        This is used if the likelihood function calls external libraries that cannot be parallelized or use
        threading in a way that conflicts with the parallelization of the non-linear search.

        Parameters
        ----------
        fitness
            The function which takes a model instance and returns its log likelihood via the Analysis class
        model
            The model which maps parameters chosen via the non-linear search (e.g. via the priors or sampling) to
            instances of the model, which are passed to the fitness function.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the search maximizes.
        checkpoint_exists
            Does the checkpoint file corresponding do a previous run of this search exist?
        """

        self.logger.info(
            """
            Running search where parallelization is disabled.
            """
        )

        search_internal = self.sampler_cls(
            prior=prior_transform,
            likelihood=fitness.__call__,
            n_dim=model.prior_count,
            prior_kwargs={"model": model},
            filepath=self.checkpoint_file,
            pool=None,
            **self.config_dict_search,
        )

        if checkpoint_exists:
            self.output_sampler_results(search_internal=search_internal)

            self.perform_update(
                model=model,
                analysis=analysis,
                during_analysis=True,
                search_internal=search_internal,
            )

        search_internal.run(
            **self.config_dict_run,
        )

        self.output_sampler_results(search_internal=search_internal)

        return search_internal

    def fit_multiprocessing(self, fitness, model, analysis, checkpoint_exists: bool):
        """
        Perform the non-linear search, using multiple CPU cores parallelized via Python's multiprocessing module.

        This uses PyAutoFit's sneaky pool class, which allows us to use the multiprocessing module in a way that plays
        nicely with the non-linear search (e.g. exception handling, keyboard interupts, etc.).

        Multiprocessing parallelization can only parallelize across multiple cores on a single device, it cannot be
        distributed across multiple devices or computing nodes. For that, use the `fit_mpi` method.

        Parameters
        ----------
        fitness
            The function which takes a model instance and returns its log likelihood via the Analysis class
        model
            The model which maps parameters chosen via the non-linear search (e.g. via the priors or sampling) to
            instances of the model, which are passed to the fitness function.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the search maximizes.
        checkpoint_exists
            Does the checkpoint file corresponding do a previous run of this search exist?
        """

        search_internal = self.sampler_cls(
            prior=prior_transform,
            likelihood=fitness.__call__,
            n_dim=model.prior_count,
            prior_kwargs={"model": model},
            filepath=self.checkpoint_file,
            pool=self.number_of_cores,
            **self.config_dict_search,
        )

        if checkpoint_exists:
            self.output_sampler_results(search_internal=search_internal)

            self.perform_update(
                model=model,
                analysis=analysis,
                during_analysis=True,
                search_internal=search_internal,
            )

        search_internal.run(
            **self.config_dict_run,
        )

        self.output_sampler_results(search_internal=search_internal)

        return search_internal

    def fit_mpi(self, fitness, model, analysis, checkpoint_exists: bool):
        """
        Perform the non-linear search, using MPI to distribute the model-fit across multiple computing nodes.

        This uses PyAutoFit's sneaky pool class, which allows us to use the multiprocessing module in a way that plays
        nicely with the non-linear search (e.g. exception handling, keyboard interupts, etc.).

        MPI parallelization can be distributed across multiple devices or computing nodes.

        Parameters
        ----------
        fitness
            The function which takes a model instance and returns its log likelihood via the Analysis class
        model
            The model which maps parameters chosen via the non-linear search (e.g. via the priors or sampling) to
            instances of the model, which are passed to the fitness function.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the search maximizes.
        checkpoint_exists
            Does the checkpoint file corresponding do a previous run of this search exist?
        """
        with self.make_sneakier_pool(
            fitness_function=fitness.__call__,
            prior_transform=prior_transform,
            fitness_args=(model, fitness.__call__),
            prior_transform_args=(model,),
        ) as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            search_internal = self.sampler_cls(
                prior=pool.prior_transform,
                likelihood=pool.fitness,
                n_dim=model.prior_count,
                filepath=self.checkpoint_file,
                pool=pool,
                **self.config_dict_search,
            )

            if checkpoint_exists:
                if self.is_master:
                    self.output_sampler_results(search_internal=search_internal)

                    self.perform_update(
                        model=model,
                        analysis=analysis,
                        during_analysis=True,
                        search_internal=search_internal,
                    )

            search_internal.run(
                **self.config_dict_run,
            )

            self.output_sampler_results(search_internal=search_internal)

        return search_internal

    def output_sampler_results(self, search_internal):
        """
        Output the sampler results to hard-disk in a generalized PyAutoFit format.

        The results in this format are loaded by other functions in order to create a `Samples` object, perform updates
        which visualize the results and write the results to the hard-disk as an output of the model-fit.

        Parameters
        ----------
        sampler
            The nautilus sampler object containing the results of the model-fit.
        """

        parameters, log_weights, log_likelihoods = search_internal.posterior()

        parameter_lists = parameters.tolist()
        log_likelihood_list = log_likelihoods.tolist()
        weight_list = np.exp(log_weights).tolist()

        search_internal = {
            "parameter_lists": parameter_lists,
            "log_likelihood_list": log_likelihood_list,
            "weight_list": weight_list,
            "log_evidence": search_internal.evidence(),
            "total_samples": int(search_internal.n_like),
            "time": self.timer.time if self.timer else None,
            "number_live_points": int(search_internal.n_live),
        }

        self.paths.save_search_internal(
            obj=search_internal,
        )

    def samples_info_from(self, search_internal=None):
        search_internal_dict = search_internal or self.paths.load_search_internal()

        if search_internal is not None:
            return {
                "log_evidence": search_internal.evidence(),
                "total_samples": int(search_internal.n_like),
                "time": self.timer.time if self.timer else None,
                "number_live_points": int(search_internal.n_live),
            }

        return {
            "log_evidence": search_internal_dict["log_evidence"],
            "total_samples": search_internal_dict["total_samples"],
            "time": self.timer.time if self.timer else None,
            "number_live_points": search_internal_dict["number_live_points"],
        }

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ):
        """
        Returns a `Samples` object from the ultranest internal results.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        if search_internal is not None:
            parameters, log_weights, log_likelihoods = search_internal.posterior()

            parameter_lists = parameters.tolist()
            log_likelihood_list = log_likelihoods.tolist()
            weight_list = np.exp(log_weights).tolist()

        else:
            search_internal_dict = self.paths.load_search_internal()

            parameter_lists = search_internal_dict["parameter_lists"]
            log_likelihood_list = search_internal_dict["log_likelihood_list"]
            weight_list = search_internal_dict["weight_list"]

        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector))
            for vector in parameter_lists
        ]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return SamplesNest(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info_from(search_internal=search_internal),
            search_internal=None,
        )

    @property
    def config_dict(self):
        return conf.instance["non_linear"]["nest"][self.__class__.__name__]

    def config_dict_with_test_mode_settings_from(self, config_dict):
        return {
            **config_dict,
            "max_iters": 1,
            "max_ncalls": 1,
        }

    def plot_results(self, samples):
        from autofit.non_linear.search.nest.nautilus.plotter import NautilusPlotter

        if not samples.pdf_converged:
            return

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["nautilus"][name]

        plotter = NautilusPlotter(
            samples=samples,
            output=Output(path=self.paths.image_path / "search", format="png"),
        )

        if should_plot("cornerplot"):
            plotter.cornerplot(
                panelsize=3.5,
                yticksize=16,
                xticksize=16,
                bins=20,
                plot_datapoints=False,
                plot_density=False,
                fill_contours=True,
                levels=(0.68, 0.95),
                labelpad=0.02,
                range=np.ones(samples.model.total_free_parameters) * 0.999,
                label_kwargs={"fontsize": 24},
            )
