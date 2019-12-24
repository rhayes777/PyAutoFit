import logging

import os
import numpy as np
import emcee

import math

from autofit import conf, exc
from autofit.optimize.non_linear.output import MCMCOutput
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.non_linear import persistent_timer

logger = logging.getLogger(__name__)


class Emcee(NonLinearOptimizer):
    def __init__(self, paths, sigma_limit=3):
        """
        Class to setup and run a MultiNest lens and output the MultiNest nlo.

        This interfaces with an input model_mapper, which is used for setting up the \
        individual model instances that are passed to each iteration of MultiNest.
        """

        super().__init__(paths)

        self.sigma_limit = sigma_limit

        self.nwalkers = conf.instance.non_linear.get("Emcee", "nwalkers", int)

        self.nsteps = conf.instance.non_linear.get("Emcee", "nsteps", int)

        logger.debug("Creating Emcee NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma_limit = self.sigma_limit
        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(
            self, paths, analysis, instance_from_physical_vector, output_results
        ):
            super().__init__(paths, analysis, output_results)
            self.instance_from_physical_vector = instance_from_physical_vector
            self.accepted_samples = 0

        def fit_instance(self, instance):
            likelihood = self.analysis.fit(instance)

            if likelihood > self.max_likelihood:

                self.max_likelihood = likelihood
                self.result = Result(instance, likelihood)

                if self.should_visualize():
                    self.analysis.visualize(instance, during_analysis=True)

                if self.should_backup():
                    self.paths.backup()

                if self.should_output_model_results():
                    self.output_results(during_analysis=True)

            return likelihood

        def __call__(self, params):

            try:

                instance = self.instance_from_physical_vector(params)
                likelihood = self.fit_instance(instance)

            except exc.FitException:

                likelihood = -np.inf

            return likelihood

    @persistent_timer
    def fit(self, analysis, model):

        output = EmceeOutput(model=model, paths=self.paths)

        fitness_function = Emcee.Fitness(
            paths=self.paths,
            analysis=analysis,
            instance_from_physical_vector=model.instance_from_physical_vector,
            output_results=output.output_results,
        )

        emcee_sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=model.prior_count,
            log_prob_fn=fitness_function.__call__,
            backend=emcee.backends.HDFBackend(filename=self.paths.path + "/emcee.hdf"),
        )

        output.save_model_info()

        try:
            emcee_state = emcee_sampler.get_last_sample()

        except AttributeError:

            emcee_state = np.zeros(shape=(emcee_sampler.nwalkers, emcee_sampler.ndim))

            for walker_index in range(emcee_sampler.nwalkers):

                emcee_state[walker_index, :] = np.asarray(
                    model.random_physical_vector_from_priors
                )

        logger.info("Running Emcee Sampling...")

        # This will be useful to testing convergence
        old_tau = np.inf

        if self.nsteps - emcee_sampler.iteration > 0:

            for sample in emcee_sampler.sample(
                initial_state=emcee_state,
                iterations=self.nsteps - emcee_sampler.iteration,
                progress=True,
                skip_initial_state_check=True,
                store=True,
            ):

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = emcee_sampler.get_autocorr_time(tol=0)

                # Check convergence

                converged = np.all(tau * 100 < emcee_sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

                if converged:
                    break
                old_tau = tau

        logger.info("Emcee complete")

        # TODO: Some of the results below use the backup_path, which isnt updated until the end if thiss function is
        # TODO: not located here. Do we need to rely just ono the optimizer foldeR? This is a good idea if we always
        # TODO: have a valid sym-link( e.g. even for aggregator use).

        self.paths.backup()

        instance = output.most_likely_model_instance

        analysis.visualize(instance=instance, during_analysis=False)
        output.output_results(during_analysis=False)
        output.output_pdf_plots()
        result = Result(
            instance=instance,
            figure_of_merit=output.maximum_log_likelihood,
            previous_model=model,
            gaussian_tuples=output.gaussian_priors_at_sigma_limit(self.sigma_limit),
        )
        self.paths.backup_zip_remove()
        return result


class EmceeOutput(MCMCOutput):
    @property
    def backend(self):
        if os.path.isfile(self.paths.sym_path + "/emcee.hdf"):
            return emcee.backends.HDFBackend(
                filename=self.paths.sym_path + "/emcee.hdf"
            )
        else:
            raise FileNotFoundError(
                "The file emcee.hdf does not exist at the path " + self.paths.path
            )

    @property
    def pdf(self):
        import getdist

        try:
            total_steps = self.backend.get_chain().shape[0]
            return getdist.mcsamples.MCSamples(
                samples=self.backend.get_chain()[: int(0.5 * total_steps) : -1, :, :]
            )
        except IOError or OSError or ValueError or IndexError:
            raise Exception

    @property
    def pdf_converged(self):
        return True

    @property
    def most_likely_index(self):
        return np.argmax(self.backend.get_log_prob(flat=True))

    @property
    def most_probable_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lens.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.

        """
        return self.pdf.getMeans()

    @property
    def most_likely_model_parameters(self):
        """
        Read the most probable or most likely model values from the 'obj_summary.txt' file which nlo from a \
        multinest lens.

        This file stores the parameters of the most probable model in the first half of entries and the most likely
        model in the second half of entries. The offset parameter is used to start at the desired model.
        """
        return self.backend.get_chain(flat=True)[self.most_likely_index]

    @property
    def maximum_log_likelihood(self):
        return self.backend.get_log_prob(flat=True)[self.most_likely_index]

    def model_parameters_at_sigma_limit(self, sigma_limit):

        limit = math.erf(0.5 * sigma_limit * math.sqrt(2))

        if self.pdf_converged:
            densities_1d = list(
                map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
            )

            return list(map(lambda p: p.getLimits(limit), densities_1d))

    @property
    def total_samples(self):
        return len(self.backend.get_log_prob(flat=True))

    def sample_model_parameters_from_sample_index(self, sample_index):
        """From a sample return the model parameters.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return list(self.pdf.samples[sample_index])

    def sample_weight_from_sample_index(self, sample_index):
        """From a sample return the sample weight.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return self.pdf.weights[sample_index]

    def sample_likelihood_from_sample_index(self, sample_index):
        """From a sample return the likelihood.

        NOTE: GetDist reads the log likelihood from the weighted_sample.txt file (column 2), which are defined as \
        -2.0*likelihood. This routine converts these back to likelihood.

        Parameters
        -----------
        sample_index : int
            The sample index of the weighted sample to return.
        """
        return self.backend.get_log_prob(flat=True)[sample_index]

    def output_pdf_plots(self):

        pass
