import numpy as np

from autofit.plot import SamplesPlotter

class NautilusPlotter(SamplesPlotter):

    def cornerplot(self, **kwargs):
        """
        Plots the `nautilus` plot `cornerplot`, using the package `corner.py`.

        This figure plots a corner plot of the 1-D and 2-D marginalized posteriors.
        """
        import corner
        import matplotlib.pyplot as plt

        points = np.asarray(self.samples.parameter_lists)

        ndim = points.shape[1]

        fig, axes = plt.subplots(ndim, ndim, figsize=(3.5*ndim, 3.5*ndim))

        corner.corner(
            data=points,
            weights=np.exp(self.samples.weight_list),
            bins=20,
            labels=self.model.parameter_labels_with_superscripts_latex,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=True,
            levels=(0.68, 0.95),
            range=np.ones(ndim) * 0.999,
            fig=fig
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")
        self.close()
