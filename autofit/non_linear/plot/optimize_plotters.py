import matplotlib.pyplot as plt
import logging
import numpy as np

from autofit.non_linear.plot.samples_plotters import SamplesPlotter

from autofit.non_linear.plot.samples_plotters import skip_plot_in_test_mode

logger = logging.getLogger(__name__)

class OptimizePlotter(SamplesPlotter):

    @skip_plot_in_test_mode
    def log_likelihood_vs_iteration(self, **kwargs):
        """
        Plot the log likelihood of a model fit to a dataset over the course of an optimization.
        """

        log_likelihood_list = self.samples.log_likelihood_list
        iteration_list = range(len(log_likelihood_list))

        plt.figure()
        plt.plot(iteration_list, log_likelihood_list, c="k")
        plt.xlabel("Iteration")
        plt.ylabel("Log Likelihood")
        plt.title("Log Likelihood vs Iteration")
        self.output.to_figure(
            auto_filename="log_likelihood_vs_iteration",
        )
        plt.close()
