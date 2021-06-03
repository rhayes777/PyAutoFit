from autofit.plot.samples_plotters import MCMCPlotter

import numpy as np
import corner

class EmceePlotter(MCMCPlotter):

    def corner(self, **kwargs):

        corner.corner(
            data=np.asarray(self.samples.parameter_lists),
            weight_list=self.samples.weight_list,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="corner")
        self.mat_plot_1d.figure.close()

    def trajectories(self, **kwargs):

        self._plot_trajectories(
            samples=self.samples.backend.get_chain(),
            log_posterior_list=self.samples.backend.get_log_prob(),
            **kwargs
        )


    def likelihood_series(self, **kwargs):

        self._plot_likelihood_series(
            log_posterior_list = self.samples.backend.get_log_prob(),
            **kwargs
        )

    def time_series(self, **kwargs):

        self._plot_time_series(
            samples=self.samples.backend.get_chain(),
        )

