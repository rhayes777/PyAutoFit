import matplotlib.pyplot as plt
import logging
import numpy as np

from autofit.non_linear.plot.samples_plotters import SamplesPlotter

from autofit.non_linear.plot.samples_plotters import skip_plot_in_test_mode

logger = logging.getLogger(__name__)

class OptimizePlotter(SamplesPlotter):

    @skip_plot_in_test_mode
    def log_likelihood_vs_iteration(self, use_log_y : bool = False, use_last_50_percent : bool = False, **kwargs):
        """
        Plot the log likelihood of a model fit as a function of iteration number.

        For a maximum likelihood estimate, the log likelihood should increase with iteration number.

        This often produces a large dynamic range in the y-axis, such that plotting the y-axis on a log-scale can be
        useful to see the full range of values.

        Parameters
        ----------
        use_log_y
            If True, the y-axis is plotted on a log-scale.
        """

        log_likelihood_list = self.samples.log_likelihood_list
        iteration_list = range(len(log_likelihood_list))

        if use_last_50_percent:

            iteration_list = iteration_list[int(len(iteration_list) / 2) :]
            log_likelihood_list = log_likelihood_list[int(len(log_likelihood_list) / 2) :]

        plt.figure(figsize=(12, 12))

        if use_log_y:
            plt.semilogy(iteration_list, log_likelihood_list, c="k")
        else:
            plt.plot(iteration_list, log_likelihood_list, c="k")

        plt.xlabel("Iteration", fontsize=16)
        plt.ylabel("Log Likelihood", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        title = "Log Likelihood vs Iteration"

        if use_log_y:

            title += " (Log Scale)"

        if use_last_50_percent:

            title += " (Last 50 Percent)"

        plt.title("Log Likelihood vs Iteration", fontsize=24)

        filename = "log_likelihood_vs_iteration"

        if use_log_y:
            filename += "_log_y"

        if use_last_50_percent:
            filename += "_last_50_percent"

        self.output.to_figure(
            auto_filename=filename,
        )
        plt.close()
