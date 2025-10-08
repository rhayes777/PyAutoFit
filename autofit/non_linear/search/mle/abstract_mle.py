from abc import ABC

from autoconf import conf
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.samples import Samples
from autofit.non_linear.plot.mle_plotters import MLEPlotter
from autofit.non_linear.plot.output import Output


class AbstractMLE(NonLinearSearch, ABC):
    @property
    def config_type(self):
        return conf.instance["non_linear"]["mle"]

    @property
    def samples_cls(self):
        return Samples

    @property
    def plotter_cls(self):
        return MLEPlotter

    def plot_results(self, samples):

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["mle"][name]

        plotter = self.plotter_cls(
            samples=samples,
            output=Output(path=self.paths.image_path / "search", format="png"),
        )

        if should_plot("subplot_parameters"):

            plotter.subplot_parameters()
            plotter.subplot_parameters(use_log_y=True)
            plotter.subplot_parameters(use_last_50_percent=True)

        if should_plot("log_likelihood_vs_iteration"):

            plotter.log_likelihood_vs_iteration()
            plotter.log_likelihood_vs_iteration(use_log_y=True)
            plotter.log_likelihood_vs_iteration(use_last_50_percent=True)