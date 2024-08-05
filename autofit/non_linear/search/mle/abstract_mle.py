from abc import ABC
import copy
from typing import List

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.analysis import Analysis
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

    def plot_start_point(
        self,
        parameter_vector : List[float],
        model: AbstractPriorModel,
        analysis: Analysis,
    ):
        """
        Visualize the starting point of the non-linear search, using an instance of the model at the starting point
        of the maximum likelihood estimator.

        Plots are output to a folder named `image_start` in the output path, so that the starting point model
        can be compared to the final model inferred by the non-linear search.

        Parameters
        ----------
        model
            The model used by the non-linear search
        analysis
            The analysis which contains the visualization methods which plot the starting point model.

        Returns
        -------

        """

        self.logger.info(
            f"Visualizing Starting Point Model in image_start folder."
        )

        instance = model.instance_from_vector(vector=parameter_vector)
        paths = copy.copy(self.paths)
        paths.image_path_suffix = "_start"

        self.perform_visualization(
            model=model,
            analysis=analysis,
            instance=instance,
            during_analysis=False,
            paths_override=paths,
        )

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