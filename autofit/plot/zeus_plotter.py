from autofit.plot import SamplesPlotter

import zeus
import matplotlib.pyplot as plt
import numpy as np

class ZeusPlotter(SamplesPlotter):

    def corner(self, **kwargs):

        zeus.cornerplot(
            samples=self.samples.zeus_sampler.get_chain(flat=True),
            weights=self.samples.weights,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="corner")

    def time_series(self, **kwargs):

        ndim = self.samples.model.prior_count

        plt.figure(figsize=(16, 1.5 * ndim))
        for n in range(ndim):
            plt.subplot2grid((ndim, 1), (n, 0))
            plt.plot(self.samples.zeus_sampler.get_chain()[:, :, n], alpha=0.5)
        plt.tight_layout()

        self.output.to_figure(structure=None, auto_filename="time_series")