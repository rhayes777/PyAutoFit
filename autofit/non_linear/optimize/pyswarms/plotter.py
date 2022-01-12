import matplotlib.pyplot as plt
import numpy as np

from autofit.plot.samples_plotters import MCMCPlotter


class PySwarmsPlotter(MCMCPlotter):

    def contour(self, **kwargs):
        from pyswarms.utils import plotters
        plotters.plot_contour(
            pos_history=self.samples.points,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="contour")
        self.close()

    def cost_history(self, **kwargs):
        from pyswarms.utils import plotters
        plotters.plot_cost_history(
            cost_history=self.samples.log_posterior_list,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cost_history")
        self.close()

    def trajectories(self, **kwargs):

        points = self.samples.points

        fig, axes = plt.subplots(self.samples.model.prior_count, figsize=(10, 7))

        for i in range(self.samples.model.prior_count):
            ax = axes[i]
            ax.plot(np.asarray(points)[:, -1, i], self.samples.log_posterior_list, "k", alpha=0.3)
            ax.set_ylabel("Log Likelihood")
            ax.set_xlabel(self.model.parameter_labels_with_superscripts_latex[i])

        self.output.to_figure(structure=None, auto_filename="trajectories")
        self.close()

    def time_series(self, **kwargs):

        fig, axes = plt.subplots(self.samples.model.prior_count, figsize=(10, 7), sharex=True)
        points = self.samples.points

        for i in range(self.samples.model.prior_count):
            ax = axes[i]
            ax.plot(np.asarray(points)[:, -1, i], "k", alpha=0.3)
            ax.set_ylabel(self.model.parameter_labels_with_superscripts_latex[i])

        axes[-1].set_xlabel("step number")

        self.output.to_figure(structure=None, auto_filename="time_series")
        self.close()
