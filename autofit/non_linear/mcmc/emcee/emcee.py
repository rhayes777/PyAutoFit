import os
from os import path
from typing import Optional

import emcee
import numpy as np

from autoconf import conf
from autofit import exc
from autofit.database.sqlalchemy_ import sa
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.mcmc.emcee.samples import EmceeSamples
from autofit.plot import EmceePlotter
from autofit.plot.output import Output


class Emcee(AbstractMCMC):
    __identifier_fields__ = (
        "nwalkers",
    )

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            prior_passer: Optional[PriorPasser] = None,
            initializer: Optional[Initializer] = None,
            auto_correlations_settings=AutoCorrelationsSettings(),
            iterations_per_update: int = None,
            number_of_cores: int = None,
            session: Optional[sa.orm.Session] = None,
            **kwargs
    ):
        """
        An Emcee non-linear search.

        For a full description of Emcee, checkout its Github and readthedocs webpages:

        https://github.com/dfm/emcee

        https://emcee.readthedocs.io/en/stable/

        If you use `Emcee` as part of a published work, please cite the package following the instructions under the
        *Attribution* section of the GitHub page.

        Parameters
        ----------
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
        auto_correlations_settings
            Customizes and performs auto correlation calculations performed during and after the search.
        number_of_cores
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            initializer=initializer,
            auto_correlations_settings=auto_correlations_settings,
            iterations_per_update=iterations_per_update,
            session=session,
            **kwargs
        )

        self.number_of_cores = number_of_cores or self._config("parallel", "number_of_cores")

        self.logger.debug("Creating Emcee Search")

    class Fitness(AbstractMCMC.Fitness):
        def __call__(self, parameters):
            try:
                return self.figure_of_merit_from(parameter_list=parameters)
            except exc.FitException:
                return self.resample_figure_of_merit

        def figure_of_merit_from(self, parameter_list):
            """
            The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. `Emcee`
            uses the log posterior.
            """
            return self.log_posterior_from(parameter_list=parameter_list)

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None):
        """
        Fit a model using Emcee and the Analysis class which contains the data and returns the log likelihood from
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
        A result object comprising the Samples object that inclues the maximum log likelihood instance and full
        chains used by the fit.
        """
        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis
        )

        pool = self.make_sneaky_pool(
            fitness_function
        )

        emcee_sampler = emcee.EnsembleSampler(
            nwalkers=self.config_dict_search["nwalkers"],
            ndim=model.prior_count,
            log_prob_fn=fitness_function.__call__,
            backend=emcee.backends.HDFBackend(
                filename=self.backend_filename
            ),
            pool=pool,
        )

        try:

            emcee_state = emcee_sampler.get_last_sample()
            samples = self.samples_from(model=model)

            total_iterations = emcee_sampler.iteration

            if samples.converged:
                iterations_remaining = 0
            else:
                iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

                self.logger.info("Existing Emcee samples found, resuming non-linear search.")

        except AttributeError:

            unit_parameter_lists, parameter_lists, log_posterior_list = self.initializer.samples_from_model(
                total_points=emcee_sampler.nwalkers,
                model=model,
                fitness_function=fitness_function,
            )

            emcee_state = np.zeros(shape=(emcee_sampler.nwalkers, model.prior_count))

            self.logger.info("No Emcee samples found, beginning new non-linear search.")

            for index, parameters in enumerate(parameter_lists):
                emcee_state[index, :] = np.asarray(parameters)

            total_iterations = 0
            iterations_remaining = self.config_dict_run["nsteps"]

        while iterations_remaining > 0:

            if self.iterations_per_update > iterations_remaining:
                iterations = iterations_remaining
            else:
                iterations = self.iterations_per_update

            for sample in emcee_sampler.sample(
                    initial_state=emcee_state,
                    iterations=iterations,
                    progress=True,
                    skip_initial_state_check=True,
                    store=True,
            ):
                pass

            emcee_state = emcee_sampler.get_last_sample()

            total_iterations += iterations
            iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

            samples = self.perform_update(
                model=model, analysis=analysis, during_analysis=True
            )

            if self.auto_correlations_settings.check_for_convergence:
                if emcee_sampler.iteration > self.auto_correlations_settings.check_size:
                    if samples.converged:
                        iterations_remaining = 0

        self.logger.info("Emcee sampling complete.")

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None):

        return Emcee.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from,
            log_likelihood_cap=log_likelihood_cap
        )

    @property
    def backend_filename(self):
        return path.join(self.paths.samples_path, "emcee.hdf")

    @property
    def backend(self) -> emcee.backends.HDFBackend:
        """
        The `Emcee` hdf5 backend, which provides access to all samples, likelihoods, etc. of the non-linear search.

        The sampler is described in the "Results" section at https://dynesty.readthedocs.io/en/latest/quickstart.html
        """

        if os.path.isfile(self.backend_filename):
            return emcee.backends.HDFBackend(
                filename=self.backend_filename
            )
        else:
            raise FileNotFoundError(
                "The file emcee.hdf does not exist at the path "
                + self.paths.samples_path
            )

    def samples_from(self, model):

        return EmceeSamples.from_results_internal(
            model=model,
            results_internal=self.backend,
            auto_correlation_settings=self.auto_correlations_settings,
            time=self.timer.time
        )

    def plot_results(self, samples):

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["emcee"][name]

        plotter = EmceePlotter(
            samples=samples,
            output=Output(path=path.join(self.paths.image_path, "search"), format="png")
        )

        if should_plot("corner"):
            plotter.corner()

        if should_plot("trajectories"):
            plotter.trajectories()

        if should_plot("likelihood_series"):
            plotter.likelihood_series()

        if should_plot("time_series"):
            plotter.time_series()
