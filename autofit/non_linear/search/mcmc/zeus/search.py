from typing import Optional

import numpy as np

from autoconf import conf
from autofit.database.sqlalchemy_ import sa
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.search.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.search.mcmc.auto_correlations import AutoCorrelations
from autofit.non_linear.search.mcmc.zeus.plotter import ZeusPlotter
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.mcmc import SamplesMCMC
from autofit.plot.output import Output

class Zeus(AbstractMCMC):
    __identifier_fields__ = (
        "nwalkers",
        "tune",
        "tolerance",
        "patience",
        "mu",
        "light_mode",
    )

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            initializer: Optional[Initializer] = None,
            auto_correlation_settings=AutoCorrelationsSettings(),
            iterations_per_update: int = None,
            number_of_cores: int = None,
            session: Optional[sa.orm.Session] = None,
            **kwargs
    ):
        """
        An Zeus non-linear search.

        For a full description of Zeus, checkout its Github and readthedocs webpages:

        https://github.com/minaskar/zeus

        https://zeus-mcmc.readthedocs.io/en/latest/

        If you use `Zeus` as part of a published work, please cite the package following the instructions under the
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
        nwalkers
            The number of walkers in the ensemble used to sample parameter space.
        nsteps
            The number of steps that must be taken by every walker. The `NonLinearSearch` will thus run for nwalkers *
            nsteps iterations.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        auto_correlation_settings : AutoCorrelationsSettings
            Customizes and performs auto correlation calculations performed during and after the search.
        number_of_cores
            The number of cores Zeus sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        number_of_cores = number_of_cores or self._config("parallel", "number_of_cores")

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            auto_correlation_settings=auto_correlation_settings,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

        self.logger.debug("Creating Zeus Search")

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using Zeus and the Analysis class which contains the data and returns the log likelihood from
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

        try:
            import zeus
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "\n--------------------\n"
                "You are attempting to perform a model-fit using Zeus. \n\n"
                "However, the optional library Zeus (https://zeus-mcmc.readthedocs.io/en/latest/) is "
                "not installed.\n\n"
                "Install it via the command `pip install zeus==3.5.5`.\n\n"
                "----------------------"
            )

        pool = self.make_pool()

        fitness = Fitness(
            model=model,
            analysis=analysis,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf
        )

        try:

            sampler = self.paths.load_results_internal()

            state = sampler.get_last_sample()
            log_posterior_list = sampler.get_last_log_prob()

            samples = self.samples_from(model=model)

            total_iterations = sampler.iteration

            if samples.converged:
                iterations_remaining = 0
            else:
                iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

                self.logger.info(
                    "Resuming Zeus non-linear search (previous samples found)."
                )

        except FileNotFoundError:

            sampler = zeus.EnsembleSampler(
                nwalkers=self.config_dict_search["nwalkers"],
                ndim=model.prior_count,
                logprob_fn=fitness.__call__,
                pool=pool,
            )

            sampler.ncall_total = 0

            (
                unit_parameter_lists,
                parameter_lists,
                log_posterior_list,
            ) = self.initializer.samples_from_model(
                total_points=sampler.nwalkers,
                model=model,
                fitness=fitness,
                test_mode_samples=False
            )

            state = np.zeros(shape=(sampler.nwalkers, model.prior_count))

            self.logger.info(
                "Starting new Zeus non-linear search (no previous samples found)."
            )

            for index, parameters in enumerate(parameter_lists):
                state[index, :] = np.asarray(parameters)

            total_iterations = 0
            iterations_remaining = self.config_dict_run["nsteps"]

        while iterations_remaining > 0:

            if self.iterations_per_update > iterations_remaining:
                iterations = iterations_remaining
            else:
                iterations = self.iterations_per_update

            for sample in sampler.sample(
                    start=state,
                    log_prob0=log_posterior_list,
                    iterations=iterations,
                    progress=True,
            ):
                pass

            sampler.ncall_total += sampler.ncall

            self.paths.save_results_internal(obj=sampler)

            state = sampler.get_last_sample()
            log_posterior_list = sampler.get_last_log_prob()

            total_iterations += iterations
            iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

            samples = self.samples_from(model=model)

            if self.auto_correlation_settings.check_for_convergence:
                if sampler.iteration > self.auto_correlation_settings.check_size:
                    if samples.converged:
                        iterations_remaining = 0

            auto_correlation_time = zeus.AutoCorrTime(samples=sampler.get_chain())

            discard = int(3.0 * np.max(auto_correlation_time))
            thin = int(np.max(auto_correlation_time) / 2.0)
            chain = sampler.get_chain(discard=discard, thin=thin, flat=True)

            if "maxcall" in self.kwargs:
                if sampler.ncall_total > self.kwargs["maxcall"]:
                    iterations_remaining = 0

            if iterations_remaining > 0:

                self.perform_update(
                    model=model, analysis=analysis, during_analysis=True
                )

    @property
    def samples_info(self):

        results_internal = self.paths.load_results_internal()

        return {
            "check_size": self.auto_correlations.check_size,
            "required_length": self.auto_correlations.required_length,
            "change_threshold": self.auto_correlations.change_threshold,
            "total_walkers": len(results_internal.get_chain()[0, :, 0]),
            "total_steps": int(results_internal.ncall_total),
            "time": self.timer.time,
        }

    def samples_via_internal_from(self, model):
        """
        Returns a `Samples` object from the zeus internal results.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        results_internal = self.paths.load_results_internal()

        discard = int(3.0 * np.max(self.auto_correlations.times))
        thin = int(np.max(self.auto_correlations.times) / 2.0)
        samples_after_burn_in =  results_internal.get_chain(discard=discard, thin=thin, flat=True)

        parameter_lists = samples_after_burn_in.tolist()
        log_posterior_list = results_internal.get_log_prob(flat=True).tolist()
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)

        log_likelihood_list = [
            log_posterior - log_prior
            for log_posterior, log_prior
            in zip(log_posterior_list, log_prior_list)
        ]

        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return SamplesMCMC(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info,
            results_internal=results_internal,
            auto_correlation_settings=self.auto_correlation_settings,
            auto_correlations=self.auto_correlations,
        )

    @property
    def auto_correlations(self):

        import zeus

        results_internal = self.paths.load_results_internal()

        times = zeus.AutoCorrTime(samples=results_internal.get_chain())
        try:
            previous_auto_correlation_times = zeus.AutoCorrTime(
                samples=results_internal.get_chain()[: - self.auto_correlation_settings.check_size, :, :],
            )
        except IndexError:
            self.logger.debug(
                "Unable to compute previous auto correlation times."
            )
            previous_auto_correlation_times = None

        return AutoCorrelations(
            check_size=self.auto_correlation_settings.check_size,
            required_length=self.auto_correlation_settings.required_length,
            change_threshold=self.auto_correlation_settings.change_threshold,
            times=times,
            previous_times=previous_auto_correlation_times,
        )

    def config_dict_with_test_mode_settings_from(self, config_dict):

        return {
            **config_dict,
            "nwalkers": 20,
            "nsteps": 10,
        }

    def plot_results(self, samples):
        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["zeus"][name]

        plotter = ZeusPlotter(
            samples=samples,
            output=Output(
                path=self.paths.image_path / "search", format="png"
            ),
        )

        if should_plot("corner"):
            plotter.corner()

        if should_plot("trajectories"):
            plotter.trajectories()

        if should_plot("likelihood_series"):
            plotter.likelihood_series()

        if should_plot("time_series"):
            plotter.time_series()