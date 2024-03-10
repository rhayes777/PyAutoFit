import matplotlib.pyplot as plt
import logging

from autofit.non_linear.plot.samples_plotters import SamplesPlotter

logger = logging.getLogger(__name__)

class MCMCPlotter(SamplesPlotter):

    def _plot_trajectories(self, samples, log_posterior_list, **kwargs):

        fig, axes = plt.subplots(self.samples.model.prior_count, figsize=(10, 7))

        for i in range(self.samples.model.prior_count):

            for walker_index in range(log_posterior_list.shape[1]):

                ax = axes[i]
                ax.plot(samples[:, walker_index, i], log_posterior_list[:, walker_index], alpha=0.3)

            ax.set_ylabel("Log Likelihood")
            ax.set_xlabel(self.model.parameter_labels_with_superscripts_latex[i])

        self.output.to_figure(structure=None, auto_filename="tracjectories")
        self.close()

    def _plot_likelihood_series(self, log_posterior_list, **kwargs):

        fig, axes = plt.subplots(1, figsize=(10, 7))

        for walker_index in range(log_posterior_list.shape[1]):

            axes.plot(log_posterior_list[:, walker_index], alpha=0.3)

        axes.set_ylabel("Log Likelihood")
        axes.set_xlabel("step number")

        self.output.to_figure(structure=None, auto_filename="likelihood_series")
        self.close()

    def _plot_time_series(self, samples, **kwargs):

        fig, axes = plt.subplots(self.samples.model.prior_count, figsize=(10, 7), sharex=True)

        for i in range(self.samples.model.prior_count):
            ax = axes[i]
            ax.plot(samples[:, :, i], alpha=0.3)
            ax.set_ylabel(self.model.parameter_labels_with_superscripts_latex[i])

        axes[-1].set_xlabel("step number")

        self.output.to_figure(structure=None, auto_filename="time_series")
        self.close()