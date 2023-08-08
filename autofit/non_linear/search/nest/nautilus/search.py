import numpy as np
import logging
import os
from typing import Optional

from autofit.database.sqlalchemy_ import sa

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.search.nest import abstract_nest
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.nest import SamplesNest
from autofit.plot import NautilusPlotter
from autofit.plot.output import Output

logger = logging.getLogger(__name__)

def prior_transform(cube, model):
    return model.vector_from_unit_vector(
        unit_vector=cube,
        ignore_prior_limits=True
    )

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
        "n_eff"
    )

    def __init__(
            self,
            name: str = "",
            path_prefix: str = "",
            unique_tag: Optional[str] = None,
            iterations_per_update: int = None,
            number_of_cores: int = None,
            session: Optional[sa.orm.Session] = None,
            mpi = False,
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
            **kwargs
        )

        self.mpi = mpi

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

        try:
            from nautilus import Sampler
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "\n--------------------\n"
                "You are attempting to perform a model-fit using Nautilus. \n\n"
                "However, the optional library Nautilus (https://nautilus-sampler.readthedocs.io/en/stable/index.html) is "
                "not installed.\n\n"
                "Install it via the command `pip install nautilus-sampler==0.7.2`.\n\n"
                "----------------------"
            )

        fitness = Fitness(
            model=model,
            analysis=analysis,
            fom_is_log_likelihood=True,
            resample_figure_of_merit=-1.0e99
        )

        if conf.instance["non_linear"]["nest"][self.__class__.__name__][
            "parallel"
        ].get("force_x1_cpu") or self.kwargs.get("force_x1_cpu"):
            pool = None
        else:
            pool = self.number_of_cores

        checkpoint_file = self.paths.search_internal_path / "checkpoint.hdf5"

        if os.path.exists(checkpoint_file):
            self.logger.info(
                "Resuming Nautilus non-linear search (previous samples found)."
            )

            self.perform_update(model=model, analysis=analysis, during_analysis=True)

        else:
            self.logger.info(
                "Starting new Nautilus non-linear search (no previous samples found)."
            )

        if conf.instance["non_linear"]["nest"][self.__class__.__name__][
            "parallel"
        ].get("force_x1_cpu") or self.kwargs.get("force_x1_cpu"):

            sampler = Sampler(
                prior=prior_transform,
                likelihood=fitness.__call__,
                n_dim=model.prior_count,
                prior_kwargs={"model": model},
                filepath=checkpoint_file,
                pool=None,
                **self.config_dict_search
            )

            sampler.run(
                **self.config_dict_run,
            )

        elif not self.mpi:

            sampler = Sampler(
                prior=prior_transform,
                likelihood=fitness.__call__,
                n_dim=model.prior_count,
                prior_kwargs={"model": model},
                filepath=checkpoint_file,
                pool=self.number_of_cores,
                **self.config_dict_search
            )

            sampler.run(
                **self.config_dict_run,
            )

        elif self.mpi:

            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            logger.info(f"Search beginning with MPI {comm.Get_rank()} / {self.number_of_cores}")

            from mpi4py.futures import MPIPoolExecutor
            pool = MPIPoolExecutor(self.number_of_cores)

            sampler = Sampler(
                prior=prior_transform,
                likelihood=fitness.__call__,
                n_dim=model.prior_count,
                prior_kwargs={"model": model},
                filepath=checkpoint_file,
                pool=pool,
                **self.config_dict_search
            )

            sampler.run(
                **self.config_dict_run,
            )

#        logger.info(f"Search ending with MPI {comm.Get_rank()} / {self.number_of_cores}")

        parameters, log_weights, log_likelihoods = sampler.posterior()

        parameter_lists = parameters.tolist()
        log_likelihood_list = log_likelihoods.tolist()
        weight_list = np.exp(log_weights).tolist()

        results_internal_json = {}

        results_internal_json["parameter_lists"] = parameter_lists
        results_internal_json["log_likelihood_list"] = log_likelihood_list
        results_internal_json["weight_list"] = weight_list
        results_internal_json["log_evidence"] = sampler.evidence()
        results_internal_json["total_samples"] = int(sampler.n_like)
        results_internal_json["time"] = self.timer.time
        results_internal_json["number_live_points"] = int(sampler.n_live)

        self.paths.save_results_internal_json(results_internal_dict=results_internal_json)

        os.remove(checkpoint_file)

        # TODO : Need max iter input (https://github.com/johannesulf/nautilus/issues/23)

        # finished = False
        #
        # while not finished:
        #
        #     try:
        #         total_iterations = sampler.ncall
        #     except AttributeError:
        #         total_iterations = 0
        #
        #     if self.config_dict_run["max_ncalls"] is not None:
        #         iterations = self.config_dict_run["max_ncalls"]
        #     else:
        #         iterations = total_iterations + self.iterations_per_update
        #
        #     if iterations > 0:
        #
        #         filter_list = ["max_ncalls", "dkl", "lepsilon"]
        #         config_dict_run = {
        #             key: value for key, value
        #             in self.config_dict_run.items()
        #             if key
        #             not in filter_list
        #         }
        #
        #         config_dict_run["update_interval_ncall"] = iterations
        #
        #         sampler.run(
        #             max_ncalls=iterations,
        #             **config_dict_run
        #         )
        #
        #     self.paths.save_results_internal(obj=sampler.results)
        #
        #     self.perform_update(model=model, analysis=analysis, during_analysis=True)
        #
        #     iterations_after_run = sampler.ncall
        #
        #     if (
        #             total_iterations == iterations_after_run
        #             or iterations_after_run == self.config_dict_run["max_ncalls"]
        #     ):
        #         finished = True

    @property
    def samples_info(self):

        results_internal_dict = self.paths.load_results_internal_json()

        return {
            "log_evidence": results_internal_dict["log_evidence"],
            "total_samples": results_internal_dict["total_samples"],
            "time": self.timer.time,
            "number_live_points": results_internal_dict["number_live_points"]
        }

    def samples_via_internal_from(self, model: AbstractPriorModel):
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

        results_internal_dict = self.paths.load_results_internal_json()

        parameter_lists = results_internal_dict["parameter_lists"]
        log_likelihood_list = results_internal_dict["log_likelihood_list"]
        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameter_lists
        ]
        weight_list = results_internal_dict["weight_list"]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return SamplesNest(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info,
            results_internal=None,
        )

    def config_dict_with_test_mode_settings_from(self, config_dict):

        return {
            **config_dict,
            "max_iters": 1,
            "max_ncalls": 1,
        }