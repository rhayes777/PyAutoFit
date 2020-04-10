import logging
import os

from autofit import conf
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.output import AbstractOutput
from autofit.optimize.non_linear.paths import Paths

logger = logging.getLogger(__name__)


class NestedSampler(NonLinearOptimizer):
    def __init__(self, paths=None, sigma=3):
        """
        Class to setup and run a MultiNest lens and output the MultiNest nlo.

        This interfaces with an input model_mapper, which is used for setting up the \
        individual model instances that are passed to each iteration of MultiNest.
        """

        if paths is None:
            paths = Paths()

        super().__init__(paths)

        self.sigma = sigma

        self.terminate_at_acceptance_ratio = self.config(
             "terminate_at_acceptance_ratio", bool
        )
        self.acceptance_ratio_threshold = self.config(
            "acceptance_ratio_threshold", float
        )

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma = self.sigma
        copy.terminate_at_acceptance_ratio = self.terminate_at_acceptance_ratio
        copy.acceptance_ratio_threshold = self.acceptance_ratio_threshold
        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(self, paths, analysis, output, terminate_at_acceptance_ratio,
                     acceptance_ratio_threshold):
            super().__init__(paths, analysis, output.output_results)
            self.accepted_samples = 0
            self.output = output

            self.model_results_output_interval = conf.instance.general.get(
                "output", "model_results_output_interval", int
            )
            self.terminate_at_acceptance_ratio = terminate_at_acceptance_ratio
            self.acceptance_ratio_threshold = acceptance_ratio_threshold

            self.terminate_has_begun = False

        def __call__(self, instance):
            if self.terminate_at_acceptance_ratio:
                if os.path.isfile(self.paths.file_summary):
                    try:
                        if (
                                self.output.acceptance_ratio < self.acceptance_ratio_threshold) or self.terminate_has_begun:
                            self.terminate_has_begun = True
                            return self.max_likelihood
                    except ValueError:
                        pass

            return self.fit_instance(instance)

    def _fit(self, analysis, model):

        output = self.output_from_model(model=model, paths=self.paths)

        output.save_model_info()

        if not os.path.exists(self.paths.has_completed_path):
            fitness_function = NestedSampler.Fitness(
                self.paths,
                analysis,
                output,
                self.terminate_at_acceptance_ratio,
                self.acceptance_ratio_threshold
            )

            logger.info("Running Nested Sampler...")
            self._simple_fit(
                model,
                fitness_function.__call__
            )
            logger.info("Nested Sampler complete")

            # TODO: Some of the results below use the backup_path, which isnt updated until the end if thiss function is
            # TODO: not located here. Do we need to rely just ono the optimizer foldeR? This is a good idea if we always
            # TODO: have a valid sym-link( e.g. even for aggregator use).

            self.paths.backup()
            open(self.paths.has_completed_path, "w+").close()
        else:
            logger.warning(
                f"{self.paths.phase_name} has run previously - skipping"
            )

        instance = output.most_likely_instance
        analysis.visualize(instance=instance, during_analysis=False)
        output.output_results(during_analysis=False)
        output.output_pdf_plots()
        result = Result(
            instance=instance,
            likelihood=output.maximum_log_likelihood,
            output=output,
            previous_model=model,
            gaussian_tuples=output.gaussian_priors_at_sigma(self.sigma),
        )
        self.paths.backup_zip_remove()
        return result

    def output_from_model(self, model, paths):
        return NestedSamplerOutput(model=model, paths=paths)


class NestedSamplerOutput(AbstractOutput):
    def weight_from_sample_index(self, sample_index):
        raise NotImplementedError()

    @property
    def evidence(self):
        raise NotImplementedError()

    def output_pdf_plots(self):

        import getdist.plots
        import matplotlib

        backend = conf.instance.visualize_general.get("general", "backend", str)
        if not backend in "default":
            matplotlib.use(backend)
        import matplotlib.pyplot as plt

        pdf_plot = getdist.plots.GetDistPlotter()

        plot_pdf_1d_params = conf.instance.visualize_plots.get("pdf", "1d_params", bool)

        if plot_pdf_1d_params:

            for param_name in self.model.param_names:
                pdf_plot.plot_1d(roots=self.pdf, param=param_name)
                pdf_plot.export(
                    fname="{}/pdf_{}_1D.png".format(self.paths.pdf_path, param_name)
                )

        plt.close()

        plot_pdf_triangle = conf.instance.visualize_plots.get("pdf", "triangle", bool)

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

