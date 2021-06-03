from autofit.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import matplotlib.pyplot as plt
import numpy as np

from autofit.plot.mat_wrap.wrap import wrap_1d
from autofit.plot.mat_wrap import visuals as vis

from typing import Optional


class AutoLabels:
    def __init__(
        self, title=None, ylabel=None, xlabel=None, legend=None, filename=None
    ):

        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.legend = legend
        self.filename = filename


class AbstractMatPlot:
    def __init__(
        self,
        units: wrap_base.Units = wrap_base.Units(),
        figure: wrap_base.Figure = wrap_base.Figure(),
        axis: wrap_base.Axis = wrap_base.Axis(),
        cmap: wrap_base.Cmap = wrap_base.Cmap(),
        colorbar: wrap_base.Colorbar = wrap_base.Colorbar(),
        colorbar_tickparams: wrap_base.ColorbarTickParams = wrap_base.ColorbarTickParams(),
        tickparams: wrap_base.TickParams = wrap_base.TickParams(),
        yticks: wrap_base.YTicks = wrap_base.YTicks(),
        xticks: wrap_base.XTicks = wrap_base.XTicks(),
        title: wrap_base.Title = wrap_base.Title(),
        ylabel: wrap_base.YLabel = wrap_base.YLabel(),
        xlabel: wrap_base.XLabel = wrap_base.XLabel(),
        legend: wrap_base.Legend = wrap_base.Legend(),
        output: wrap_base.Output = wrap_base.Output(),
    ):
        """
        Visualizes data structures (e.g an `Array2D`, `Grid2D`, `VectorField`, etc.) using Matplotlib.

        The `Plotter` is passed objects from the `wrap_base` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        The following data structures can be plotted using the following matplotlib functions:

        - `Array2D`:, using `plt.imshow`.
        - `Grid2D`: using `plt.scatter`.
        - `Line`: using `plt.plot`, `plt.semilogy`, `plt.loglog` or `plt.scatter`.
        - `VectorField`: using `plt.quiver`.
        - `RectangularMapper`: using `plt.imshow`.
        - `VoronoiMapper`: using `plt.fill`.

        Parameters
        ----------
        units : wrap_base.Units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : wrap_base.Figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`
        axis : wrap_base.Axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap : wrap_base.Cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar : wrap_base.Colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams : wrap_base.ColorbarTickParams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams : wrap_base.TickParams
            Customizes the appearances of the y and x ticks on the plot (e.g. the fontsize) using `plt.tick_params`.
        yticks : wrap_base.YTicks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks : wrap_base.XTicks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title : wrap_base.Title
            Sets the figure title and customizes its appearance using `plt.title`.
        ylabel : wrap_base.YLabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : wrap_base.XLabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : wrap_base.Legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : wrap_base.Output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        """

        self.units = units
        self.figure = figure
        self.axis = axis
        self.cmap = cmap
        self.colorbar = colorbar
        self.colorbar_tickparams = colorbar_tickparams
        self.tickparams = tickparams
        self.title = title
        self.yticks = yticks
        self.xticks = xticks
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.legend = legend
        self.output = output

        self.number_subplots = None
        self.subplot_index = None

    def set_for_subplot(self, is_for_subplot: bool):
        """
        Sets the `is_for_subplot` attribute for every `MatWrap` object in this `MatPlot` object by updating
        the `is_for_subplot`. By changing this tag:

            - The [subplot] section of the config file of every `MatWrap` object is used instead of [figure].
            - Calls which output or close the matplotlib figure are over-ridden so that the subplot is not removed.

        Parameters
        ----------
        is_for_subplot : bool
            The entry the `is_for_subplot` attribute of every `MatWrap` object is set too.
        """
        self.is_for_subplot = is_for_subplot
        self.output.bypass = is_for_subplot

        for attr, value in self.__dict__.items():
            if hasattr(value, "is_for_subplot"):
                value.is_for_subplot = is_for_subplot

    def get_subplot_rows_columns(self, number_subplots):
        """
        Get the size of a sub plotter in (total_y_pixels, total_x_pixels), based on the number of subplots that are
        going to be plotted.

        Parameters
        -----------
        number_subplots : int
            The number of subplots that are to be plotted in the figure.
        """
        if number_subplots <= 2:
            return 1, 2
        elif number_subplots <= 4:
            return 2, 2
        elif number_subplots <= 6:
            return 2, 3
        elif number_subplots <= 9:
            return 3, 3
        elif number_subplots <= 12:
            return 3, 4
        elif number_subplots <= 16:
            return 4, 4
        elif number_subplots <= 20:
            return 4, 5
        else:
            return 6, 6

    def setup_subplot(self, aspect=None, subplot_rows_columns=None):

        if subplot_rows_columns is None:
            rows, columns = self.get_subplot_rows_columns(
                number_subplots=self.number_subplots
            )
        else:
            rows = subplot_rows_columns[0]
            columns = subplot_rows_columns[1]

        if aspect is None:
            plt.subplot(rows, columns, self.subplot_index)
        else:
            plt.subplot(rows, columns, self.subplot_index, aspect=float(aspect))

        self.subplot_index += 1


class MatPlot1D(AbstractMatPlot):
    def __init__(
        self,
        units: wrap_base.Units = wrap_base.Units(),
        figure: wrap_base.Figure = wrap_base.Figure(),
        axis: wrap_base.Axis = wrap_base.Axis(),
        cmap: wrap_base.Cmap = wrap_base.Cmap(),
        colorbar: wrap_base.Colorbar = wrap_base.Colorbar(),
        colorbar_tickparams: wrap_base.ColorbarTickParams = wrap_base.ColorbarTickParams(),
        tickparams: wrap_base.TickParams = wrap_base.TickParams(),
        yticks: wrap_base.YTicks = wrap_base.YTicks(),
        xticks: wrap_base.XTicks = wrap_base.XTicks(),
        title: wrap_base.Title = wrap_base.Title(),
        ylabel: wrap_base.YLabel = wrap_base.YLabel(),
        xlabel: wrap_base.XLabel = wrap_base.XLabel(),
        legend: wrap_base.Legend = wrap_base.Legend(),
        output: wrap_base.Output = wrap_base.Output(),
        yx_plot: wrap_1d.YXPlot = wrap_1d.YXPlot(),
        vertical_line_axvline: wrap_1d.AXVLine = wrap_1d.AXVLine(),
    ):
        """
        Visualizes 1D data structures (e.g a `Line`, etc.) using Matplotlib.

        The `Plotter` is passed objects from the `wrap_base` package which wrap matplotlib plot functions and customize
        the appearance of the plots of the data structure. If the values of these matplotlib wrapper objects are not
        manually specified, they assume the default values provided in the `config.visualize.mat_*` `.ini` config files.

        The following 1D data structures can be plotted using the following matplotlib functions:

        - `Line` using `plt.plot`.

        Parameters
        ----------
        units : wrap_base.Units
            The units of the figure used to plot the data structure which sets the y and x ticks and labels.
        figure : wrap_base.Figure
            Opens the matplotlib figure before plotting via `plt.figure` and closes it once plotting is complete
            via `plt.close`.
        axis : wrap_base.Axis
            Sets the extent of the figure axis via `plt.axis` and allows for a manual axis range.
        cmap : wrap_base.Cmap
            Customizes the colormap of the plot and its normalization via matplotlib `colors` objects such
            as `colors.Normalize` and `colors.LogNorm`.
        colorbar : wrap_base.Colorbar
            Plots the colorbar of the plot via `plt.colorbar` and customizes its tick labels and values using method
            like `cb.set_yticklabels`.
        colorbar_tickparams : wrap_base.ColorbarTickParams
            Customizes the yticks of the colorbar plotted via `plt.colorbar`.
        tickparams : wrap_base.TickParams
            Customizes the appearances of the y and x ticks on the plot, (e.g. the fontsize), using `plt.tick_params`.
        yticks : wrap_base.YTicks
            Sets the yticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.yticks`.
        xticks : wrap_base.XTicks
            Sets the xticks of the plot, including scaling them to new units depending on the `Units` object, via
            `plt.xticks`.
        title : wrap_base.Title
            Sets the figure title and customizes its appearance using `plt.title`.
        ylabel : wrap_base.YLabel
            Sets the figure ylabel and customizes its appearance using `plt.ylabel`.
        xlabel : wrap_base.XLabel
            Sets the figure xlabel and customizes its appearance using `plt.xlabel`.
        legend : wrap_base.Legend
            Sets whether the plot inclues a legend and customizes its appearance and labels using `plt.legend`.
        output : wrap_base.Output
            Sets if the figure is displayed on the user's screen or output to `.png` using `plt.show` and `plt.savefig`
        yx_plot : wrap_1d.YXPlot
            Sets how the y versus x plot appears, for example if it each axis is linear or log, using `plt.plot`.
        vertical_line_axvline : wrao_1d
            Sets how a vertical line plotted on the figure using the `plt.axvline` method.
        """

        super().__init__(
            units=units,
            figure=figure,
            axis=axis,
            cmap=cmap,
            colorbar=colorbar,
            colorbar_tickparams=colorbar_tickparams,
            tickparams=tickparams,
            yticks=yticks,
            xticks=xticks,
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            legend=legend,
            output=output,
        )

        self.yx_plot = yx_plot
        self.vertical_line_axvline = vertical_line_axvline

        self.is_for_multi_plot = False
        self.is_for_subplot = False

    def set_for_multi_plot(self, is_for_multi_plot: bool, color: str):
        """
        Sets the `is_for_subplot` attribute for every `MatWrap` object in this `MatPlot` object by updating
        the `is_for_subplot`. By changing this tag:

            - The [subplot] section of the config file of every `MatWrap` object is used instead of [figure].
            - Calls which output or close the matplotlib figure are over-ridden so that the subplot is not removed.

        Parameters
        ----------
        is_for_subplot : bool
            The entry the `is_for_subplot` attribute of every `MatWrap` object is set too.
        """
        self.is_for_multi_plot = is_for_multi_plot
        self.output.bypass = is_for_multi_plot

        self.yx_plot.kwargs["c"] = color
        self.vertical_line_axvline.kwargs["c"] = color

        self.vertical_line_axvline.no_label = True

    def plot_yx(
        self,
        y,
        x,
        visuals_1d: vis.Visuals1D,
        auto_labels: AutoLabels,
        plot_axis_type_override: Optional[str] = None,
        bypass: bool = False,
    ):

        if y is None:
            return

        if (not self.is_for_subplot) and (not self.is_for_multi_plot):
            self.figure.open()
        else:
            if not bypass:
                if self.is_for_subplot:
                    self.setup_subplot()

        self.title.set(auto_title=auto_labels.title)

        if x is None:
            x = np.arange(len(y))

        if self.yx_plot.plot_axis_type is None:
            plot_axis_type = "linear"
        else:
            plot_axis_type = self.yx_plot.plot_axis_type

        if plot_axis_type_override is not None:
            plot_axis_type = plot_axis_type_override

        self.yx_plot.plot_y_vs_x(
            y=y,
            x=x,
            label=auto_labels.legend,
            plot_axis_type=plot_axis_type,
        )

        self.ylabel.set(units=self.units, include_brackets=False)
        self.xlabel.set(units=self.units, include_brackets=False)

        self.tickparams.set()

        if plot_axis_type == "symlog":
            plt.yscale('symlog')

        self.xticks.set(
            array=None, min_value=np.min(x), max_value=np.max(x), units=self.units
        )

        self.title.set(auto_title=auto_labels.title)
        self.ylabel.set(units=self.units, auto_label=auto_labels.ylabel)
        self.xlabel.set(units=self.units, auto_label=auto_labels.xlabel)

        visuals_1d.plot_via_plotter(plotter=self)

        if auto_labels.legend is not None:  # or vertical_line_labels is not None:
            self.legend.set()

        if (not self.is_for_subplot) and (not self.is_for_multi_plot):
            self.output.to_figure(structure=None, auto_filename=auto_labels.filename)
            self.figure.close()


