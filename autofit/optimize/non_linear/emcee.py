import logging

import os
import numpy as np
import emcee

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

        self.nwalkers = conf.instance.non_linear.get(
            "Emcee", "nwalkers", int
        )

        self.nsteps = conf.instance.non_linear.get(
            "Emcee", "nsteps", int
        )

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

            # if likelihood > self.max_likelihood:
            #
            #     self.max_likelihood = likelihood
            #     self.result = Result(instance, likelihood)
            #
            #     if self.should_visualize():
            #         self.analysis.visualize(instance, during_analysis=True)
            #
            #     if self.should_backup():
            #         self.paths.backup()
            #
            #     if self.should_output_model_results():
            #         self.output_results(during_analysis=True)

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

        fitness_function = Emcee.Fitness(
            paths=self.paths,
            analysis=analysis,
            instance_from_physical_vector=model.instance_from_physical_vector,
            output_results=None,
        )

        emcee_sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=model.prior_count,
            log_prob_fn=fitness_function.__call__,
            backend=emcee.backends.HDFBackend(filename=self.paths.path + "/emcee")
        )

        output = EmceeOutput(model=model, paths=self.paths)

        output.save_model_info()

        try:
            emcee_state = emcee_sampler.get_last_sample()

        except AttributeError:

            emcee_state = np.zeros(shape=(emcee_sampler.nwalkers, emcee_sampler.ndim))

            for walker_index in range(emcee_sampler.nwalkers):

                emcee_state[walker_index, :] = np.asarray(model.random_physical_vector_from_priors)

        logger.info("Running Emcee Sampling...")

        # This will be useful to testing convergence
        old_tau = np.inf

        if self.nsteps - emcee_sampler.iteration > 0:

            for sample in emcee_sampler.sample(
                initial_state=emcee_state,
                iterations=self.nsteps - emcee_sampler.iteration,
                progress=True,
                skip_initial_state_check=True,
                store=True
            ):

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = emcee_sampler.get_autocorr_time(tol=0)

                # Check convergence

                converged = np.all(tau * 100 < emcee_sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
             #   if converged:
             #       break
                old_tau = tau

        logger.info("Emcee complete")

        # TODO: Some of the results below use the backup_path, which isnt updated until the end if thiss function is
        # TODO: not located here. Do we need to rely just ono the optimizer foldeR? This is a good idea if we always
        # TODO: have a valid sym-link( e.g. even for aggregator use).

        self.paths.backup()

        instance = output.most_likely_model_instance

        stop

    #    analysis.visualize(instance=instance, during_analysis=False)
    #    output.output_results(during_analysis=False)
    #    output.output_pdf_plots()
    #     result = Result(
    #         instance=instance,
    #         figure_of_merit=output.evidence,
    #         previous_model=model,
    #         gaussian_tuples=output.gaussian_priors_at_sigma_limit(
    #             self.sigma_limit
    #         ),
    #     )
    #     self.paths.backup_zip_remove()
        return None


class EmceeOutput(MCMCOutput):

    @property
    def backend(self):
        if os.path.isfile(self.paths.sym_path + "/emcee.hdf"):
            return emcee.backends.HDFBackend(filename=self.paths.sym_path + "/emcee.hdf")
        else:
            raise FileNotFoundError("The file emcee.hdf does not exist at the path " + self.paths.path)

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
        return self.read_list_of_results_from_summary_file(
            number_entries=self.model.prior_count, offset=0
        )

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

    @property
    def evidence(self):
        return self.read_list_of_results_from_summary_file(
            number_entries=2, offset=112
        )[0]

    def read_list_of_results_from_summary_file(self, number_entries, offset):

        summary = open(self.paths.file_summary)
        summary.read(2 + offset * self.model.prior_count)
        vector = []
        for param in range(number_entries):
            vector.append(float(summary.read(28)))

        summary.close()

        return vector

    def model_parameters_at_sigma_limit(self, sigma_limit):
        limit = math.erf(0.5 * sigma_limit * math.sqrt(2))
        densities_1d = list(
            map(lambda p: self.pdf.get1DDensity(p), self.pdf.getParamNames().names)
        )
        return list(map(lambda p: p.getLimits(limit), densities_1d))

    @property
    def total_samples(self):
        return len(self.pdf.weights)

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
        return -0.5 * self.pdf.loglikes[sample_index]

    def output_pdf_plots(self):

        import getdist.plots
        import matplotlib

        backend = conf.instance.visualize.get("figures", "backend", str)
        matplotlib.use(backend)
        import matplotlib.pyplot as plt

        pdf_plot = getdist.plots.GetDistPlotter()

        plot_pdf_1d_params = conf.instance.visualize.get(
            "plots", "plot_pdf_1d_params", bool
        )

        if plot_pdf_1d_params:

            for param_name in self.model.param_names:
                pdf_plot.plot_1d(roots=self.pdf, param=param_name)
                pdf_plot.export(
                    fname="{}/pdf_{}_1D.png".format(self.paths.pdf_path, param_name)
                )

        plt.close()

        plot_pdf_triangle = conf.instance.visualize.get(
            "plots", "plot_pdf_triangle", bool
        )

        if plot_pdf_triangle:

            try:
                pdf_plot.triangle_plot(roots=self.pdf)
                pdf_plot.export(fname="{}/pdf_triangle.png".format(self.paths.pdf_path))
            except Exception as e:
                print(type(e))
                print(
                    "The PDF triangle of this non-linear search could not be plotted. This is most likely due to a "
                    "lack of smoothness in the sampling of parameter space. Sampler further by decreasing the "
                    "parameter evidence_tolerance."
                )

        plt.close()

    def output_results(self, during_analysis):

        if os.path.isfile(self.paths.file_summary):

            results = []

            results += text_util.label_and_value_string(
                label="Bayesian Evidence ",
                value=self.evidence,
                whitespace=90,
                format_string="{:.8f}",
            )
            results += ["\n"]
            results += text_util.label_and_value_string(
                label="Maximum Likelihood ",
                value=self.maximum_log_likelihood,
                whitespace=90,
                format_string="{:.8f}",
            )
            results += ["\n\n"]

            results += ["Most Likely Model:\n\n"]
            most_likely = self.most_likely_model_parameters

            formatter = text_formatter.TextFormatter()

            for i, prior_path in enumerate(self.model.unique_prior_paths):
                formatter.add((prior_path, self.format_str.format(most_likely[i])))
            results += [formatter.text + "\n"]

            if not during_analysis:

                results += self.results_from_sigma_limit(limit=3.0)
                results += ["\n"]
                results += self.results_from_sigma_limit(limit=1.0)

                results += ["\n\ninstances\n"]

                formatter = text_formatter.TextFormatter()

                for t in self.model.path_float_tuples:
                    formatter.add(t)

                results += ["\n" + formatter.text]

            text_util.output_list_of_strings_to_file(
                file=self.paths.file_results, list_of_strings=results
            )