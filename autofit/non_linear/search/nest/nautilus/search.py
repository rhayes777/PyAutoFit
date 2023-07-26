import copy
from os import path
from typing import Optional

from autofit.database.sqlalchemy_ import sa

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.nest import abstract_nest
from autofit.non_linear.search.nest.abstract_nest import AbstractNest
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.nest import SamplesNest
from autofit.plot import NautilusPlotter
from autofit.plot.output import Output

class Nautilus(abstract_nest.AbstractNest):
    __identifier_fields__ = (
        "n_live",
    )

    def __init__(
            self,
            name: str = "",
            path_prefix: str = "",
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

        For a full description of Nautilus and its Python wrapper PyNautilus, checkout its Github and documentation
        webpages:

        https://github.com/JohannesBuchner/Nautilus
        https://johannesbuchner.github.io/Nautilus/readme.html

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
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
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

        self.logger.debug("Creating Nautilus Search")

    class Fitness(AbstractNest.Fitness):
        @property
        def resample_figure_of_merit(self):
            """
            If a sample raises a FitException, this value is returned to signify that the point requires resampling or
            should be given a likelihood so low that it is discard.

            -np.inf is an invalid sample value for Dynesty, so we instead use a large negative number.
            """
            return -1.0e99

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None):
        """
        Fit a model using Dynesty and the Analysis class which contains the data and returns the log likelihood from
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
                "However, the optional library Nautilus (https://johannesbuchner.github.io/Nautilus/index.html) is "
                "not installed.\n\n"
                "Install it via the command `pip install ultranest==3.6.2`.\n\n"
                "----------------------"
            )

        fitness = self.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from,
            log_likelihood_cap=log_likelihood_cap,
        )

        def prior_transform(cube):
            return model.vector_from_unit_vector(
                unit_vector=cube,
                ignore_prior_limits=True
            )

        self.sampler = Sampler(
            prior=prior_transform,
            likelihood=fitness.__call__,
            n_dim=model.prior_count,
            pool=None,
            filepath=self.paths.search_internal_path / "checkpoint.hdf5",
            **self.config_dict_search
        )

        self.sampler.run(verbose=True)

        self.perform_update(model=model, analysis=analysis, during_analysis=True)

        return

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

        results_internal = self.sampler

        return {
            "log_evidence": results_internal.evidence(),
            "total_samples": int(results_internal.n_like),
            "time": self.timer.time,
            "number_live_points": int(results_internal.n_live)
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

        results_internal = self.sampler

        parameters, weights, log_likelihood_list = results_internal.posterior()

        parameters = parameters
        log_likelihood_list = log_likelihood_list
        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameters
        ]
        weight_list = weights

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return SamplesNest(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info,
            results_internal=results_internal,
        )

    def config_dict_with_test_mode_settings_from(self, config_dict):

        return {
            **config_dict,
            "max_iters": 1,
            "max_ncalls": 1,
        }