from autofit.plot import SamplesPlotter

from dynesty import plotting as dyplot

import matplotlib.pyplot as plt


class DynestyPlotter(SamplesPlotter):

    def cornerplot(self, **kwargs):

        dyplot.cornerplot(
            results=self.samples.results,
            labels=self.model.parameter_labels_latex,
            **kwargs
        )

        self.output.to_figure(structure=None)