from os import path
from typing import List, Optional
from sqlalchemy.orm import Session
import zeus
import numpy as np

from autoconf import conf

from autofit import exc
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.log import logger
from autofit.non_linear.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelationsSettings, AutoCorrelations
from autofit.non_linear.samples import MCMCSamples, Sample

from autofit.plot import ZeusPlotter
from autofit.plot.mat_wrap.wrap.wrap_base import Output

class Zeus(AbstractMCMC):

    __identifier_fields__ = (
        "nwalkers",
        "tune",
        "tolerance",
        "patience",
        "mu",
        "light_mode"
    )

    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag : Optional[str] = None,
            prior_passer=None,
            initializer=None,
            auto_correlations_settings=AutoCorrelationsSettings(),
            iterations_per_update : int = None,
            number_of_cores : int = None,
            session : Optional[Session] = None,
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
        prior_passer
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        nwalkers : int
            The number of walkers in the ensemble used to sample parameter space.
        nsteps : int
            The number of steps that must be taken by every walker. The `NonLinearSearch` will thus run for nwalkers *
            nsteps iterations.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        auto_correlations_settings : AutoCorrelationsSettings
            Customizes and performs auto correlation calculations performed during and after the search.
        number_of_cores : int
            The number of cores Zeus sampling is performed using a Python multiprocessing Pool instance. If 1, a
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

        logger.debug("Creating Zeus Search")

    class Fitness(AbstractMCMC.Fitness):
        def __call__(self, parameters):
            try:
                return self.figure_of_merit_from(parameter_list=parameters)
            except exc.FitException:
                return self.resample_figure_of_merit

        def figure_of_merit_from(self, parameter_list):
            """
            The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. 
            
            `Zeus` uses the log posterior.
            """
            return self.log_posterior_from(parameter_list=parameter_list)

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None):
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

        pool = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis
        )

        if self.paths.is_object("zeus"):

            zeus_sampler = self.zeus_pickled

            zeus_state = zeus_sampler.get_last_sample()
            initial_log_posterior_list = zeus_sampler.get_last_log_prob()

            samples = self.samples_from(model=model)

            total_iterations = zeus_sampler.iteration

            if samples.converged:
                iterations_remaining = 0
            else:
                iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

                logger.info("Existing Zeus samples found, resuming non-linear search.")

        else:

            zeus_sampler = zeus.EnsembleSampler(
                nwalkers=self.config_dict_search["nwalkers"],
                ndim=model.prior_count,
                logprob_fn=fitness_function.__call__,
                pool=pool,
            )

            zeus_sampler.ncall_total = 0

            initial_unit_parameter_lists, initial_parameter_lists, initial_log_posterior_list = self.initializer.initial_samples_from_model(
                total_points=zeus_sampler.nwalkers,
                model=model,
                fitness_function=fitness_function,
            )

            zeus_state = np.zeros(shape=(zeus_sampler.nwalkers, model.prior_count))

            logger.info("No Zeus samples found, beginning new non-linear search.")

            for index, parameters in enumerate(initial_parameter_lists):

                zeus_state[index, :] = np.asarray(parameters)

            total_iterations = 0
            iterations_remaining = self.config_dict_run["nsteps"]

        while iterations_remaining > 0:

            if self.iterations_per_update > iterations_remaining:
                iterations = iterations_remaining
            else:
                iterations = self.iterations_per_update

            for sample in zeus_sampler.sample(
                    start=zeus_state,
                    log_prob0=initial_log_posterior_list,
                    iterations=iterations,
                    progress=True,
            ):

                pass

            zeus_sampler.ncall_total += zeus_sampler.ncall

            self.paths.save_object(
                "zeus",
                zeus_sampler
            )

            zeus_state = zeus_sampler.get_last_sample()
            initial_log_posterior_list = zeus_sampler.get_last_log_prob()

            total_iterations += iterations
            iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

            samples = self.perform_update(
                model=model, analysis=analysis, during_analysis=True
            )

            if self.auto_correlations_settings.check_for_convergence:
                if zeus_sampler.iteration > self.auto_correlations_settings.check_size:
                    if samples.converged:
                        iterations_remaining = 0

            auto_correlation_time = zeus.AutoCorrTime(samples=zeus_sampler.get_chain())

            discard = int(3.0 * np.max(auto_correlation_time))
            thin = int(np.max(auto_correlation_time) / 2.0)
            chain = zeus_sampler.get_chain(discard=discard, thin=thin, flat=True)

            if "maxcall" in self.kwargs:
                if zeus_sampler.ncall_total > self.kwargs["maxcall"]:
                    iterations_remaining = 0

        logger.info("Zeus sampling complete.")

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None):

        return Zeus.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from,
            log_likelihood_cap=log_likelihood_cap
        )

    def samples_from(self, model):
        """Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the `NonLinearSearch` chains,
            etc.
        """

        return ZeusSamples(
            model=model,
            zeus_sampler=self.zeus_pickled,
            auto_correlation_settings=self.auto_correlations_settings,
            time=self.timer.time
        )

    @property
    def zeus_pickled(self):
        return self.paths.load_object(
            "zeus"
        )

    def plot_results(self, samples):

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["emcee"][name]

        plotter = ZeusPlotter(
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

class ZeusSamples(MCMCSamples):

    def __init__(
            self,
            model: ModelMapper,
            zeus_sampler,
            auto_correlation_settings: AutoCorrelationsSettings,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):
        """
        Attributes
        ----------
        total_walkers : int
            The total number of walkers used by this MCMC non-linear search.
        total_steps : int
            The total number of steps taken by each walker of this MCMC `NonLinearSearch` (the total samples is equal
            to the total steps * total walkers).
        """

        self.zeus_sampler = zeus_sampler

        super().__init__(
            model=model,
            auto_correlation_settings=auto_correlation_settings,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

        self._samples = None

    @property
    def samples(self):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the `NonLinearSearch` chains,
            etc.
        """

        if self._samples is not None:
            return self._samples

        parameter_lists = self.zeus_sampler.get_chain(flat=True).tolist()
        log_prior_list = [
            sum(self.model.log_prior_list_from_vector(vector=vector)) for vector in parameter_lists
        ]
        log_posterior_list = self.zeus_sampler.get_log_prob(flat=True).tolist()
        log_likelihood_list = [log_posterior - log_prior for log_posterior, log_prior in zip(log_posterior_list, log_prior_list)]

        weight_list = len(log_likelihood_list) * [1.0]

        self._samples = Sample.from_lists(
            model=self.model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return self._samples

    @property
    def samples_after_burn_in(self) -> [list]:
        """The zeus samples with the initial burn-in samples removed.

        The burn-in period is estimated using the auto-correlation times of the parameters."""

        discard = int(3.0 * np.max(self.auto_correlations.times))
        thin = int(np.max(self.auto_correlations.times) / 2.0)
        return self.zeus_sampler.get_chain(discard=discard, thin=thin, flat=True)

    @property
    def total_walkers(self):
        return len(self.zeus_sampler.get_chain()[0, :, 0])

    @property
    def total_steps(self):
        return int(self.zeus_sampler.ncall_total)

    @property
    def auto_correlations(self):

        times = zeus.AutoCorrTime(samples=self.zeus_sampler.get_chain())
        try:
            previous_auto_correlation_times = zeus.AutoCorrTime(
                samples=self.zeus_sampler.get_chain()[: -self.auto_correlation_settings.check_size, :, :],
            )
        except IndexError:
            previous_auto_correlation_times = None

        return AutoCorrelations(
            check_size=self.auto_correlation_settings.check_size,
            required_length=self.auto_correlation_settings.required_length,
            change_threshold=self.auto_correlation_settings.change_threshold,
            times=times,
            previous_times=previous_auto_correlation_times,
        )
