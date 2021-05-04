from typing import List, Optional
import zeus
import numpy as np

from autofit import exc
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.log import logger
from autofit.non_linear.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelationsSettings, AutoCorrelations
from autofit.non_linear.samples import MCMCSamples, Sample

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
            session : Optional[bool] = None,
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
        initializer : non_linear.initializer.Initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        auto_correlations_settings : AutoCorrelationsSettings
            Customizes and performs auto correlation calculations performed during and after the search.
        number_of_cores : int
            The number of cores Zeus sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.

        All remaining attributes are zeus parameters and described at the zeus API webpage:

        https://zeus-mcmc.readthedocs.io/en/latest/
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
                return self.figure_of_merit_from_parameters(parameters=parameters)
            except exc.FitException:
                return self.resample_figure_of_merit

        def figure_of_merit_from_parameters(self, parameters):
            """
            The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. 
            
            `Zeus` uses the log posterior.
            """
            try:
                return self.log_posterior_from_parameters(parameters=parameters)
            except exc.FitException:
                raise exc.FitException

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

        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis, pool_ids=pool_ids
        )

        if self.paths.is_object("zeus"):

            zeus_sampler = self.zeus_pickled

            zeus_state = zeus_sampler.get_last_sample()
            initial_log_posteriors = zeus_sampler.get_last_log_prob()

            samples = self.samples_via_sampler_from_model(model=model)

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

            initial_unit_parameters, initial_parameters, initial_log_posteriors = self.initializer.initial_samples_from_model(
                total_points=zeus_sampler.nwalkers,
                model=model,
                fitness_function=fitness_function,
            )

            zeus_state = np.zeros(shape=(zeus_sampler.nwalkers, model.prior_count))

            logger.info("No Zeus samples found, beginning new non-linear search.")

            for index, parameters in enumerate(initial_parameters):

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
                    log_prob0=initial_log_posteriors,
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
            initial_log_posteriors = zeus_sampler.get_last_log_prob()

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

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None, pool_ids=None):

        return Zeus.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_via_sampler_from_model,
            log_likelihood_cap=log_likelihood_cap,
            pool_ids=pool_ids,
        )

    def samples_via_sampler_from_model(self, model):
        """Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Zeus, all quantities are extracted via the hdf5 backend of results.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the `NonLinearSearch` chains,
            etc.
        """

        zeus_sampler = self.zeus_pickled

        parameters = zeus_sampler.get_chain(flat=True).tolist()
        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]
        log_likelihoods = zeus_sampler.get_log_prob(flat=True).tolist()
        weights = len(log_likelihoods) * [1.0]
        total_walkers = len(zeus_sampler.get_chain()[0, :, 0])
        total_steps = int(zeus_sampler.ncall_total)

        auto_correlation_time = zeus.AutoCorrTime(samples=zeus_sampler.get_chain())
        try:
            previous_auto_correlation_times = zeus.AutoCorrTime(
                samples=zeus_sampler.get_chain()[: -self.auto_correlations_settings.check_size, :, :],
            )
        except IndexError:
            previous_auto_correlation_times = None

        return ZeusSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=log_likelihoods,
                log_priors=log_priors,
                weights=weights
            ),
            paths=self.paths,
            auto_correlations=AutoCorrelations(
                check_size=self.auto_correlations_settings.check_size,
                required_length=self.auto_correlations_settings.required_length,
                change_threshold=self.auto_correlations_settings.change_threshold,
                times=auto_correlation_time,
                previous_times=previous_auto_correlation_times
            ),
            total_walkers=total_walkers,
            total_steps=total_steps,
            time=self.timer.time
        )

    def samples_via_csv_json_from_model(self, model):

        zeus_sampler = self.zeus_pickled

        # TODO : Better design to remove repetition.

        samples = self.paths.load_samples()
        samples_info = self.paths.load_samples_info()

        try:
            auto_correlation_times = zeus.AutoCorrTime(samples=zeus_sampler.get_chain())
            previous_auto_correlation_times = zeus.AutoCorrTime(
                samples=zeus_sampler.get_chain()[: -self.auto_correlations_settings.check_size, :, :],
            )
        except FileNotFoundError:
            auto_correlation_times = None
            previous_auto_correlation_times = None

        return ZeusSamples(
            model=model,
            samples=samples,
            paths=self.paths,
            auto_correlations=AutoCorrelations(
                check_size=samples_info["check_size"],
                required_length=samples_info["required_length"],
                change_threshold=samples_info["change_threshold"],
                times=auto_correlation_times,
                previous_times=previous_auto_correlation_times,
            ),
            total_walkers=samples_info["total_walkers"],
            total_steps=samples_info["total_steps"],
            time=samples_info["time"],
        )

    @property
    def zeus_pickled(self):
        return self.paths.load_object(
            "zeus"
        )

class ZeusSamples(MCMCSamples):

    def __init__(
            self,
            model: ModelMapper,
            samples: List[Sample],
            auto_correlations: AutoCorrelations,
            total_walkers: int,
            total_steps: int,
            paths,
            unconverged_sample_size: int = 100,
            time: float = None,
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

        super().__init__(
            model=model,
            samples=samples,
            auto_correlations=auto_correlations,
            total_walkers=total_walkers,
            total_steps=total_steps,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

        self.paths = paths

    @property
    def samples_after_burn_in(self) -> [list]:
        """The zeus samples with the initial burn-in samples removed.

        The burn-in period is estimated using the auto-correlation times of the parameters."""

        zeus_sampler = self.paths.load_object(
            "zeus"
        )

        discard = int(3.0 * np.max(self.auto_correlations.times))
        thin = int(np.max(self.auto_correlations.times) / 2.0)
        chain = zeus_sampler.get_chain(discard=discard, thin=thin, flat=True)

        return chain
