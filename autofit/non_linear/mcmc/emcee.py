import json
import os
from typing import List

import emcee
import numpy as np

from autofit import exc
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear import samples as samp
from autofit.non_linear.log import logger
from autofit.non_linear.mcmc.abstract_mcmc import AbstractMCMC
from autofit.non_linear.paths import convert_paths
from autofit.non_linear.samples import MCMCSamples, Sample


class Emcee(AbstractMCMC):

    @convert_paths
    def __init__(
            self,
            paths=None,
            prior_passer=None,
            nwalkers=None,
            nsteps=None,
            initializer=None,
            auto_correlation_check_for_convergence=None,
            auto_correlation_check_size=None,
            auto_correlation_required_length=None,
            auto_correlation_change_threshold=None,
            iterations_per_update=None,
            number_of_cores=None,
    ):
        """ An Emcee non-linear search.

        For a full description of Emcee, checkout its Github and readthedocs webpages:

        https://github.com/dfm/emcee

        https://emcee.readthedocs.io/en/stable/

        Extensions:

        - Provides the option to check the auto-correlation length of the samples during the run and terminating
          sampling early if these meet a specified threshold. See this page
          (https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr) for a description.

        - Provides different options for walker initialization, with the default 'ball' method starting all walkers
          close to one another in parameter space, as recommended in the Emcee documentation
          (https://emcee.readthedocs.io/en/stable/user/faq/).

        If you use *Emcee* as part of a published work, please cite the package following the instructions under the
        *Attribution* section of the GitHub page.

        Parameters
        ----------
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        nwalkers : int
            The number of walkers in the ensemble used to sample parameter space.
        nsteps : int
            The number of steps that must be taken by every walker. The `NonLinearSearch` will thus run for nwalkers *
            nsteps iterations.
        initializer : non_linear.initializer.Initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        auto_correlation_check_for_convergence : bool
            Whether the auto-correlation lengths of the Emcee samples are checked to determine the stopping criteria.
            If `True`, this option may terminate the Emcee run before the input number of steps, nsteps, has
            been performed. If `False` nstep samples will be taken.
        auto_correlation_check_size : int
            The length of the samples used to check the auto-correlation lengths (from the latest sample backwards).
            For convergence, the auto-correlations must not change over a certain range of samples. A longer check-size
            thus requires more samples meet the auto-correlation threshold, taking longer to terminate sampling.
            However, shorter chains risk stopping sampling early due to noise.
        auto_correlation_required_length : int
            The length an auto_correlation chain must be for it to be used to evaluate whether its change threshold is
            sufficiently small to terminate sampling early.
        auto_correlation_change_threshold : float
            The threshold value by which if the change in auto_correlations is below sampling will be terminated early.
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.

        All remaining attributes are emcee parameters and described at the emcee API webpage:

        https://emcee.readthedocs.io/en/stable/
        """

        self.nwalkers = (
            self._config("search", "nwalkers") if nwalkers is None else nwalkers
        )
        self.nsteps = (
            self._config("search", "nsteps") if nsteps is None else nsteps
        )

        self.auto_correlation_check_for_convergence = (
            self._config("auto_correlation", "check_for_convergence")
            if auto_correlation_check_for_convergence is None
            else auto_correlation_check_for_convergence
        )
        self.auto_correlation_check_size = (
            self._config("auto_correlation", "check_size")
            if auto_correlation_check_size is None
            else auto_correlation_check_size
        )
        self.auto_correlation_required_length = (
            self._config("auto_correlation", "required_length")
            if auto_correlation_required_length is None
            else auto_correlation_required_length
        )
        self.auto_correlation_change_threshold = (
            self._config("auto_correlation", "change_threshold")
            if auto_correlation_change_threshold is None
            else auto_correlation_change_threshold
        )

        super().__init__(
            paths=paths,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
        )

        self.number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        logger.debug("Creating Emcee NLO")

    class Fitness(AbstractMCMC.Fitness):
        def __call__(self, parameters):
            try:
                return self.figure_of_merit_from_parameters(parameters=parameters)
            except exc.FitException:
                return self.resample_figure_of_merit

        def figure_of_merit_from_parameters(self, parameters):
            """The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. *Emcee*
            uses the log posterior.
            """
            try:
                return self.log_posterior_from_parameters(parameters=parameters)
            except exc.FitException:
                raise exc.FitException

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

        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis, pool_ids=pool_ids
        )

        emcee_sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=model.prior_count,
            log_prob_fn=fitness_function.__call__,
            backend=emcee.backends.HDFBackend(
                filename=self.paths.samples_path + "/emcee.hdf"
            ),
            pool=pool,
        )

        try:

            emcee_state = emcee_sampler.get_last_sample()
            samples = self.samples_via_sampler_from_model(model=model)

            total_iterations = emcee_sampler.iteration

            if samples.converged:
                iterations_remaining = 0
            else:
                iterations_remaining = self.nsteps - total_iterations

                logger.info("Existing Emcee samples found, resuming non-linear search.")

        except AttributeError:

            initial_unit_parameters, initial_parameters, initial_log_posteriors = self.initializer.initial_samples_from_model(
                total_points=emcee_sampler.nwalkers,
                model=model,
                fitness_function=fitness_function,
            )

            emcee_state = np.zeros(shape=(emcee_sampler.nwalkers, model.prior_count))

            logger.info("No Emcee samples found, beginning new non-linear search.")

            for index, parameters in enumerate(initial_parameters):

                emcee_state[index, :] = np.asarray(parameters)

            total_iterations = 0
            iterations_remaining = self.nsteps

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
            iterations_remaining = self.nsteps - total_iterations

            samples = self.perform_update(
                model=model, analysis=analysis, during_analysis=True
            )

            if emcee_sampler.iteration % self.auto_correlation_check_size:
                if samples.converged and self.auto_correlation_check_for_convergence:
                    iterations_remaining = 0

        logger.info("Emcee sampling complete.")

    @property
    def tag(self):
        """Tag the output folder of the PySwarms non-linear search, according to the number of particles and
        parameters defining the search strategy."""

        name_tag = self._config("tag", "name")
        nwalkers_tag = f"{self._config('tag', 'nwalkers')}_{self.nwalkers}"

        return f"{name_tag}[{nwalkers_tag}]"

    def copy_with_name_extension(self, extension, path_prefix=None, remove_phase_tag=False):
        """Copy this instance of the emcee `NonLinearSearch` with all associated attributes.

        This is used to set up the `NonLinearSearch` on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, path_prefix=path_prefix, remove_phase_tag=remove_phase_tag
        )
        copy.prior_passer = self.prior_passer
        copy.nwalkers = self.nwalkers
        copy.nsteps = self.nsteps
        copy.auto_correlation_check_for_convergence = (
            self.auto_correlation_check_for_convergence
        )
        copy.auto_correlation_check_size = self.auto_correlation_check_size
        copy.auto_correlation_required_length = self.auto_correlation_required_length
        copy.auto_correlation_change_threshold = self.auto_correlation_change_threshold
        copy.initializer = self.initializer
        copy.iterations_per_update = self.iterations_per_update
        copy.number_of_cores = self.number_of_cores

        return copy

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None, pool_ids=None):

        return Emcee.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_via_sampler_from_model,
            log_likelihood_cap=log_likelihood_cap,
            pool_ids=pool_ids,
        )

    def samples_via_sampler_from_model(self, model):
        """Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Emcee, all quantities are extracted via the hdf5 backend of results.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the `NonLinearSearch` chains,
            etc.
        """

        parameters = self.backend.get_chain(flat=True).tolist()
        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]
        log_likelihoods = self.backend.get_log_prob(flat=True).tolist()
        weights = len(log_likelihoods) * [1.0]
        auto_correlation_time = self.backend.get_autocorr_time(tol=0)
        total_walkers = len(self.backend.get_chain()[0, :, 0])
        total_steps = len(self.backend.get_log_prob())

        return EmceeSamples(
            model=model,
            samples=Sample.from_lists(
                model=model,
                parameters=parameters,
                log_likelihoods=log_likelihoods,
                log_priors=log_priors,
                weights=weights
            ),
            total_walkers=total_walkers,
            total_steps=total_steps,
            auto_correlation_times=auto_correlation_time,
            auto_correlation_check_size=self.auto_correlation_check_size,
            auto_correlation_required_length=self.auto_correlation_required_length,
            auto_correlation_change_threshold=self.auto_correlation_change_threshold,
            backend=self.backend,
            time=self.timer.time,
        )

    def samples_via_csv_json_from_model(self, model):

        # TODO : Better design to remove repetition.

        samples = samp.load_from_table(
            filename=self.paths.samples_file
        )

        with open(self.paths.info_file) as infile:
            samples_info = json.load(infile)

        return EmceeSamples(
            model=model,
            samples=samples,
            auto_correlation_times=self.backend.get_autocorr_time(tol=0),
            auto_correlation_check_size=samples_info["auto_correlation_check_size"],
            auto_correlation_required_length=samples_info[
                "auto_correlation_required_length"
            ],
            auto_correlation_change_threshold=samples_info[
                "auto_correlation_change_threshold"
            ],
            total_walkers=samples_info["total_walkers"],
            total_steps=samples_info["total_steps"],
            time=samples_info["time"],
            backend=self.backend
        )

    @property
    def backend(self) -> emcee.backends.HDFBackend:
        """The *Emcee* hdf5 backend, which provides access to all samples, likelihoods, etc. of the non-linear search.

        The sampler is described in the "Results" section at https://dynesty.readthedocs.io/en/latest/quickstart.html"""
        if os.path.isfile(self.paths.samples_path + "/emcee.hdf"):
            return emcee.backends.HDFBackend(
                filename=self.paths.samples_path + "/emcee.hdf"
            )
        else:
            raise FileNotFoundError(
                "The file emcee.hdf does not exist at the path "
                + self.paths.samples_path
            )


class EmceeSamples(MCMCSamples):

    def __init__(
            self,
            model: ModelMapper,
            samples: List[Sample],
            auto_correlation_times: np.ndarray,
            auto_correlation_check_size: int,
            auto_correlation_required_length: int,
            auto_correlation_change_threshold: float,
            total_walkers: int,
            total_steps: int,
            backend: emcee.backends.HDFBackend,
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
            auto_correlation_times=auto_correlation_times,
            auto_correlation_check_size=auto_correlation_check_size,
            auto_correlation_required_length=auto_correlation_required_length,
            auto_correlation_change_threshold=auto_correlation_change_threshold,
            total_walkers=total_walkers,
            total_steps=total_steps,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

        self.backend = backend

    @property
    def samples_after_burn_in(self) -> [list]:
        """The emcee samples with the initial burn-in samples removed.

        The burn-in period is estimated using the auto-correlation times of the parameters."""
        discard = int(3.0 * np.max(self.auto_correlation_times))
        thin = int(np.max(self.auto_correlation_times) / 2.0)
        return self.backend.get_chain(discard=discard, thin=thin, flat=True)

    @property
    def previous_auto_correlation_times(self) -> [float]:
        return emcee.autocorr.integrated_time(
            x=self.backend.get_chain()[: -self.auto_correlation_check_size, :, :], tol=0
        )
