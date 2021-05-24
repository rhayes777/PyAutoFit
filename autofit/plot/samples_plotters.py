from autofit.plot.mat_wrap.wrap import wrap_base
from autofit.plot.mat_wrap import visuals as vis
from autofit.plot.mat_wrap import include as inc
from autofit.plot.mat_wrap import mat_plot as mp
from autofit.plot import abstract_plotters

import matplotlib.pyplot as plt

class SamplesPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
            self, 
            samples,
            mat_plot_1d: mp.MatPlot1D = mp.MatPlot1D(),
            visuals_1d: vis.Visuals1D = vis.Visuals1D(),
            include_1d: inc.Include1D = inc.Include1D(),
            output : wrap_base.Output = wrap_base.Output()
    ):

        self.samples = samples
        self.output = output

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
        )

    @property
    def model(self):
        return self.samples.model

    @property
    def visuals_with_include_2d(self):

        return self.visuals_2d + self.visuals_2d.__class__()


class MCMCPlotter(SamplesPlotter):

    def _plot_trajectories(self, samples, log_posterior_list, **kwargs):

        fig, axes = plt.subplots(self.samples.model.prior_count, figsize=(10, 7))

        for i in range(self.samples.model.prior_count):

            for walker_index in range(log_posterior_list.shape[1]):

                ax = axes[i]
                ax.plot_array(samples[:, walker_index, i], log_posterior_list[:, walker_index], alpha=0.3)

            ax.set_ylabel("Log Likelihood")
            ax.set_xlabel(self.model.parameter_labels_latex[i])

        self.output.to_figure(structure=None, auto_filename="tracjectories")
        self.mat_plot_1d.figure.close()

    def _plot_likelihood_series(self, log_posterior_list, **kwargs):

        fig, axes = plt.subplots(1, figsize=(10, 7))

        for walker_index in range(log_posterior_list.shape[1]):

            axes.plot_array(log_posterior_list[:, walker_index], alpha=0.3)

        axes.set_ylabel("Log Likelihood")
        axes.set_xlabel("step number")

        self.output.to_figure(structure=None, auto_filename="likelihood_series")
        self.mat_plot_1d.figure.close()

    def _plot_time_series(self, samples, **kwargs):

        fig, axes = plt.subplots(self.samples.model.prior_count, figsize=(10, 7), sharex=True)

        for i in range(self.samples.model.prior_count):
            ax = axes[i]
            ax.plot_array(samples[:, :, i], alpha=0.3)
            ax.set_ylabel(self.model.parameter_labels_latex[i])

        axes[-1].set_xlabel("step number")

        self.output.to_figure(structure=None, auto_filename="time_series")
        self.mat_plot_1d.figure.close()