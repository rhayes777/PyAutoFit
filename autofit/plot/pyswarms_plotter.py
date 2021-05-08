from autofit.plot import SamplesPlotter
from pyswarms.utils import plotters

import numpy as np

class PySwarmsPlotter(SamplesPlotter):

    def contour(self, **kwargs):

        plotters.plot_contour(
            pos_history=self.samples.points,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="contour")

    def cost_history(self, **kwargs):

        print(self.samples.log_posteriors)

        plotters.plot_cost_history(
            cost_history=self.samples.log_posteriors,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cost_history")

    # def surface(self, **kwargs):
    #
    #     plotters.plot_surface(
    #         pos_history=self.samples.points,
    #         **kwargs
    #     )
    #
    #     self.output.to_figure(structure=None, auto_filename="surface")