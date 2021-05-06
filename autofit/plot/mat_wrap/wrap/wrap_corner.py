from autofit.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import corner

from dynesty import plotting as dyplot

class Corner(wrap_base.AbstractMatWrap):

    @property
    def config_folder(self):
        return "mat_wrap_corner"

    def __init__(self, **kwargs):
        """
        Plots the PDF as a corner plot using the library corner.py

        This object wraps the following corner.py method:

        - corner.corner: https://corner.readthedocs.io/en/latest/api.html
        """

        super().__init__(**kwargs)

    def corner(self, xs, weights, labels):

        corner.corner(
            xs=xs,
            weights=weights,
            labels=labels,
            # range=[(8.0, 11.0), (0.0, 1.0), (0.0, 1.0)],
            # truths=[0.0, 0.0, 0.0],
            truth_color='grey',
            plot_density=False,
            plot_datapoints=False,
            levels=[0.0, 0.99],
            fill_contours=False,
      #      hist_kwargs={'color': cfcolors[1], 'alpha': 1.0, 'linewidth': 2.0},
      #      contourf_kwargs={'colors': cfcolors, 'alpha': 0.5},
      #      contour_kwargs={'colors': cfcolors[1], 'alpha': 1.0, 'linewidths': 2.0}
     )