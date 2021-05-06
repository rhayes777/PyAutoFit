from autofit.plot import SamplesPlotter

from dynesty import plotting as dyplot

import matplotlib.pyplot as plt


class DynestyPlotter(SamplesPlotter):

    def figures(self, cornerplot=False):

        if cornerplot:

            plt.figure(figsize=(100, 100))
            dyplot.cornerplot(results=self.samples.results)
            plt.show()