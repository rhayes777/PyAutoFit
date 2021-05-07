from autofit.plot import SamplesPlotter

import matplotlib.pyplot as plt
import corner

class EmceePlotter(SamplesPlotter):

    def corner(self, **kwargs):

        corner.corner(
            xs=self.samples.parameters,
            weights=self.samples.weights,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="corner")

    def time_series(self, **kwargs):

        fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        samples = self.samples.backend.get_chain()

        for i in range(self.samples.model.prior_count):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.model.parameter_labels_latex[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        self.output.to_figure(structure=None, auto_filename="time_series")