from autofit.plot.samples_plotters import MCMCPlotter

import corner

class EmceePlotter(MCMCPlotter):

    def corner(self, **kwargs):

        corner.corner(
            xs=self.samples.parameters,
            weights=self.samples.weights,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="corner")

    def trajectories(self, **kwargs):

        self._plot_trajectories(
            samples=self.samples.backend.get_chain(),
            log_posteriors=self.samples.backend.get_log_prob(),
            **kwargs
        )


    def likelihood_series(self, **kwargs):

        self._plot_likelihood_series(
            log_posteriors = self.samples.backend.get_log_prob(),
            **kwargs
        )

    def time_series(self, **kwargs):

        self._plot_time_series(
            samples=self.samples.backend.get_chain(),
        )

