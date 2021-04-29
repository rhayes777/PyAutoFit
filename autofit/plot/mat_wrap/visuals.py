from autofit.plot.mat_wrap import mat_plot, include as inc
from typing import Optional
from abc import ABC


class AbstractVisuals(ABC):
    def __add__(self, other):
        """
        Adds two `Visuals` classes together.

        When we perform plotting, the `Include` class is used to create additional `Visuals` class from the data
        structures that are plotted, for example:

        mask = Mask2D.circular(shape_native=(100, 100), pixel_scales=0.1, radius=3.0)
        array = Array2D.ones(shape_native=(100, 100), pixel_scales=0.1)
        masked_array = al.Array2D.manual_mask(array=array, mask=mask)
        include_2d = Include2D(mask=True)
        array_plotter = aplt.Array2DPlotter(array=masked_array, include_2d=include_2d)
        array_plotter.figure()

        Because `mask=True` in `Include2D` the function `figure` extracts the `Mask2D` from the `masked_array`
        and plots it. It does this by creating a new `Visuals2D` object.

        If the user did not manually input a `Visuals2d` object, the one created in `function_array` is the one used to
        plot the image

        However, if the user specifies their own `Visuals2D` object and passed it to the plotter, e.g.:

        visuals_2d = Visuals2D(origin=(0.0, 0.0))
        include_2d = Include2D(mask=True)
        array_plotter = aplt.Array2DPlotter(array=masked_array, include_2d=include_2d)

        We now wish for the `Plotter` to plot the `origin` in the user's input `Visuals2D` object and the `Mask2d`
        extracted via the `Include2D`. To achieve this, two `Visuals2D` objects are created: (i) the user's input
        instance (with an origin) and; (ii) the one created by the `Include2D` object (with a mask).

        This `__add__` override means we can add the two together to make the final `Visuals2D` object that is
        plotted on the figure containing both the `origin` and `Mask2D`.:

        visuals_2d = visuals_2d_via_user + visuals_2d_via_include

        The ordering of the addition has been specifically chosen to ensure that the `visuals_2d_via_user` does not
        retain the attributes that are added to it by the `visuals_2d_via_include`. This ensures that if multiple plots
        are made, the same `visuals_2d_via_user` is used for every plot. If this were not the case, it would
        permenantly inherit attributes from the `Visuals` from the `Include` method and plot them on all figures.
        """

        for attr, value in self.__dict__.items():
            try:
                if other.__dict__[attr] is None and self.__dict__[attr] is not None:
                    other.__dict__[attr] = self.__dict__[attr]
            except KeyError:
                pass

        return other


class Visuals1D(AbstractVisuals):
    def __init__(
        self,
        vertical_line: Optional[float] = None,
    ):

        self.vertical_line = vertical_line

    @property
    def plotter(self):
        return mat_plot.MatPlot1D()

    @property
    def include(self):
        return inc.Include1D()

    def plot_via_plotter(self, plotter):

        if self.vertical_line is not None:

            plotter.vertical_line_axvline.axvline_vertical_line(
                vertical_line=self.vertical_line
            )


class Visuals2D(AbstractVisuals):
    def __init__(
        self,
    ):

        pass

    def plot_via_plotter(self, plotter):

        pass
