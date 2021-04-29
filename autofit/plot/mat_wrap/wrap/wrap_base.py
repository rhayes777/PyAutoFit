from autoconf import conf
import matplotlib

from autoarray.structures.arrays.two_d import array_2d


def set_backend():

    backend = conf.get_matplotlib_backend()

    if backend not in "default":
        matplotlib.use(backend)

    try:
        hpc_mode = conf.instance["general"]["hpc"]["hpc_mode"]
    except KeyError:
        hpc_mode = False

    if hpc_mode:
        matplotlib.use("Agg")


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
from os import path
import os
import typing


from autoarray.structures import abstract_structure
from autoarray import exc


class Units:
    def __init__(
        self,
        use_scaled: bool = None,
        conversion_factor: float = None,
        in_kpc: bool = None,
    ):
        """
        This object controls the units of a plotted figure, and performs multiple tasks when making the plot:

        1) Species the units of the plot (e.g. meters, kilometers) and contains a conversion factor which converts
           the plotted data from its current units (e.g. meters) to the units plotted (e.g. kilometeters). Pixel units
            can be used if `use_scaled=False`.

        2) Uses the conversion above to manually override the yticks and xticks of the figure, so it appears in the
           converted units.

        3) Sets the ylabel and xlabel to include a string containing the units.

        Parameters
        ----------
        use_scaled : bool
            If True, plot the 2D data with y and x ticks corresponding to its scaled coordinates (its `pixel_scales`
            attribute is used as the `conversion_factor`). If `False` plot them in pixel units.
        conversion_factor : float
            If plotting the labels in scaled units, this factor multiplies the values that are used for the labels.
            This allows for additional unit conversions of the figure labels.
        in_kpc : bool
            If True, the scaled units are converted to kilo-parsecs via the input Comsology of the plot (this is only
            relevent for the projects PyAutoGalaxy / PyAutoLens).
        """

        self.use_scaled = use_scaled
        self.conversion_factor = conversion_factor
        self.in_kpc = in_kpc

        if use_scaled is not None:
            self.use_scaled = use_scaled
        else:
            try:
                self.use_scaled = conf.instance["visualize"]["general"]["general"][
                    "use_scaled"
                ]
            except:
                self.use_scaled = True

        try:
            self.in_kpc = (
                in_kpc
                if in_kpc is not None
                else conf.instance["visualize"]["general"]["units"]["in_kpc"]
            )
        except:
            self.in_kpc = None


class AbstractMatWrap:
    def __init__(self, **kwargs):
        """
        An abstract base class for wrapping matplotlib plotting methods.
        
        Classes are used to wrap matplotlib so that the data structures in the `autoarray.structures` package can be 
        plotted in standardized withs. This exploits how these structures have specific formats, units, properties etc.
        This allows us to make a simple API for plotting structures, for example to plot an `Array2D` structure:
        
        import autoarray as aa
        import autoarray.plot as aplt
        
        arr = aa.Array2D.manual_native(array=[[1.0, 1.0], [2.0, 2.0]], pixel_scales=2.0)
        aplt.Array2D(array=arr)
        
        The wrapped Mat objects make it simple to customize how matplotlib visualizes this data structure, for example
        we can customize the figure size and colormap using the `Figure` and `Cmap` objects.
        
        figure = aplt.Figure(figsize=(7,7), aspect="square")
        cmap = aplt.Cmap(cmap="jet", vmin=1.0, vmax=2.0)
        
        plotter = aplt.MatPlot2D(figure=figure, cmap=cmap)
        
        aplt.Array2D(array=arr, plotter=plotter)
        
        The `Plotter` object is detailed in the `autoarray.plot.plotter` package.
        
        The matplotlib wrapper objects in ths module also use configuration files to choose their default settings.
        For example, in `autoarray.config.visualize.mat_base.Figure.ini` you will note the section:
        
        [figure]
        figsize=(7, 7)
        
        [subplot]
        figsize=auto
        
        This specifies that when a data structure (like the `Array2D` above) is plotted, the figsize will always be 
        (7,7) when a single figure is plotted and it will be chosen automatically if a subplot is plotted. This
        allows one to customize the matplotlib settings of every plot in a project.
        """

        self.is_for_subplot = False
        self.kwargs = kwargs

    @property
    def config_dict(self):

        if not self.is_for_subplot:

            config_dict = conf.instance["visualize"][self.config_folder][
                self.__class__.__name__
            ]["figure"]._dict

        else:

            config_dict = conf.instance["visualize"][self.config_folder][
                self.__class__.__name__
            ]["subplot"]._dict

        if "c" in config_dict:
            config_dict["c"] = remove_spaces_and_commas_from_colors(
                colors=config_dict["c"]
            )

        return {**config_dict, **self.kwargs}

    @property
    def config_folder(self):
        return "mat_wrap"


class Figure(AbstractMatWrap):
    """
    Sets up the Matplotlib figure before plotting (this is used when plotting individual figures and subplots).

    This object wraps the following Matplotlib methods:

    - plt.figure: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.figure.html
    - plt.close: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.close.html

    It also controls the aspect ratio of the figure plotted.
    """

    @property
    def config_dict(self):
        """Creates a config dict of valid inputs of the method `plt.figure` from the object's config_dict."""

        config_dict = super().config_dict

        if config_dict["figsize"] == "auto":
            config_dict["figsize"] = None
        elif isinstance(config_dict["figsize"], str):
            config_dict["figsize"] = tuple(
                map(int, config_dict["figsize"][1:-1].split(","))
            )

        return config_dict

    def aspect_for_subplot_from_grid(self, grid):

        ratio = float(
            (grid.scaled_maxima[1] - grid.scaled_minima[1])
            / (grid.scaled_maxima[0] - grid.scaled_minima[0])
        )

        if self.config_dict["aspect"] in "square":
            return ratio
        elif self.config_dict["aspect"] in "auto":
            return 1.0 / ratio
        elif self.config_dict["aspect"] in "equal":
            return 1.0

    def aspect_from_shape_native(
        self, shape_native: typing.Union[typing.Tuple[int, int]]
    ) -> typing.Union[float, str]:
        """
        Returns the aspect ratio of the figure from the 2D shape of a data structure.

        This is used to ensure that rectangular arrays are plotted as square figures on sub-plots.

        Parameters
        ----------
        shape_native : (int, int)
            The two dimensional shape of an `Array2D` that is to be plotted.
        """
        if isinstance(self.config_dict["aspect"], str):
            if self.config_dict["aspect"] in "square":
                return float(shape_native[1]) / float(shape_native[0])

        return self.config_dict["aspect"]

    def open(self):
        """Wraps the Matplotlib method 'plt.figure' for opening a figure."""
        if not plt.fignum_exists(num=1):
            config_dict = self.config_dict
            config_dict.pop("aspect")
            plt.figure(**config_dict)

    def close(self):
        """Wraps the Matplotlib method 'plt.close' for closing a figure."""
        if plt.fignum_exists(num=1):
            plt.close()


class Axis(AbstractMatWrap):
    def __init__(self, symmetric_source_centre: bool = False, **kwargs):
        """
        Customizes the axis of the plotted figure.

        This object wraps the following Matplotlib method:

         plt.axis: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.axis.html

        Parameters
        ----------
        symmetric_source_centre : bool
            If `True`, the axis is symmetric around the centre of the plotted structure's coordinates.
         """

        super().__init__(**kwargs)

        self.symmetric_around_centre = symmetric_source_centre

    def set(self, extent, grid=None):
        """
        Set the axis limits of the figure the grid is plotted on.

        Parameters
        -----------
        extent : [float, float, float, float]
            The extent of the figure which set the axis-limits on the figure the grid is plotted, following [xmin, xmax, ymin, ymax].
        """

        config_dict = self.config_dict
        extent_dict = config_dict.get("extent")

        if extent_dict is not None:
            config_dict.pop("extent")

        if self.symmetric_around_centre:

            ymin = np.min(grid[:, 0])
            ymax = np.max(grid[:, 0])
            xmin = np.min(grid[:, 1])
            xmax = np.max(grid[:, 1])

            x = np.max([np.abs(xmin), np.abs(xmax)])
            y = np.max([np.abs(ymin), np.abs(ymax)])

            extent_symmetric = [-x, x, -y, y]

            plt.axis(extent_symmetric, **config_dict)

        else:

            if extent_dict is not None:
                plt.axis(extent_dict, **config_dict)
            else:
                plt.axis(extent, **config_dict)


class Cmap(AbstractMatWrap):
    """
    Customizes the Matplotlib colormap and its normalization.

    This object wraps the following Matplotlib methods:

    - colors.Linear: https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html
    - colors.LogNorm: https://matplotlib.org/3.3.2/tutorials/colors/colormapnorms.html
    - colors.SymLogNorm: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.colors.SymLogNorm.html

    The cmap that is created is passed into various Matplotlib methods, most notably imshow:

     https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html
    """

    def vmin_from_array(self, array: np.ndarray):

        if self.config_dict["vmin"] is None:
            return np.min(array)
        else:
            return self.config_dict["vmin"]

    def vmax_from_array(self, array: np.ndarray):

        if self.config_dict["vmax"] is None:
            return np.max(array)
        else:
            return self.config_dict["vmax"]

    def norm_from_array(self, array: np.ndarray) -> object:
        """
        Returns the `Normalization` object which scales of the colormap.

        If vmin / vmax are not manually input by the user, the minimum / maximum values of the data being plotted
        are used.

        Parameters
        -----------
        array : np.ndarray
            The array of data which is to be plotted.
        """

        vmin = self.vmin_from_array(array=array)
        vmax = self.vmax_from_array(array=array)

        if self.config_dict["norm"] in "linear":
            return colors.Normalize(vmin=vmin, vmax=vmax)
        elif self.config_dict["norm"] in "log":
            if vmin == 0.0:
                vmin = 1.0e-4
            return colors.LogNorm(vmin=vmin, vmax=vmax)
        elif self.config_dict["norm"] in "symmetric_log":
            return colors.SymLogNorm(
                vmin=vmin,
                vmax=vmax,
                linthresh=self.config_dict["linthresh"],
                linscale=self.config_dict["linscale"],
            )
        else:
            raise exc.PlottingException(
                "The normalization (norm) supplied to the plotter is not a valid string (must be "
                "{linear, log, symmetric_log}"
            )


class Colorbar(AbstractMatWrap):
    def __init__(
        self,
        manual_tick_labels: typing.Optional[typing.List[float]] = None,
        manual_tick_values: typing.Optional[typing.List[float]] = None,
        **kwargs,
    ):
        """
        Customizes the colorbar of the plotted figure.

        This object wraps the following Matplotlib method:

         plt.colorbar: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.colorbar.html

        The colorbar object `cb` that is created is also customized using the following methods:

         cb.set_yticklabels: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html

        Parameters
        ----------
        manual_tick_labels : [float]
            Manually override the colorbar tick labels to an input list of float.
        manual_tick_values : [float]
            If the colorbar tick labels are manually specified the locations on the colorbar they appear running 0 -> 1.
         """

        super().__init__(**kwargs)

        self.manual_tick_labels = manual_tick_labels
        self.manual_tick_values = manual_tick_values

    def set(self):
        """ Set the figure's colorbar, optionally overriding the tick labels and values with manual inputs. """

        if self.manual_tick_values is None and self.manual_tick_labels is None:
            cb = plt.colorbar(**self.config_dict)
        elif (
            self.manual_tick_values is not None and self.manual_tick_labels is not None
        ):
            cb = plt.colorbar(ticks=self.manual_tick_values, **self.config_dict)
            cb.ax.set_yticklabels(labels=self.manual_tick_labels)
        else:
            raise exc.PlottingException(
                "Only 1 entry of tick_values or tick_labels was input. You must either supply"
                "both the values and labels, or neither."
            )

        return cb

    def set_with_color_values(self, cmap: str, color_values: np.ndarray):
        """
        Set the figure's colorbar using an array of already known color values.

        This method is used for producing the color bar on a Voronoi mesh plot, which is unable to use the in-built
        Matplotlib colorbar method.

        Parameters
        ----------
        cmap : str
            The colormap used to map normalized data values to RGBA colors (see
            https://matplotlib.org/3.3.2/api/cm_api.html).
        color_values : np.ndarray
            The values of the pixels on the Voronoi mesh which are used to create the colorbar.
        """

        cax = cm.ScalarMappable(cmap=cmap)
        cax.set_array(color_values)

        if self.manual_tick_values is None and self.manual_tick_labels is None:
            plt.colorbar(mappable=cax, **self.config_dict)
        elif (
            self.manual_tick_values is not None and self.manual_tick_labels is not None
        ):
            cb = plt.colorbar(
                mappable=cax, ticks=self.manual_tick_values, **self.config_dict
            )
            cb.ax.set_yticklabels(self.manual_tick_labels)


class ColorbarTickParams(AbstractMatWrap):
    """
    Customizes the ticks of the colorbar of the plotted figure.

    This object wraps the following Matplotlib colorbar method:

     cb.set_yticklabels: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html
     """

    def set(self, cb):

        cb.ax.tick_params(**self.config_dict)


class TickParams(AbstractMatWrap):
    """
    The settings used to customize a figure's y and x ticks parameters.

    This object wraps the following Matplotlib methods:

    - plt.tick_params: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.tick_params.html
    """

    def set(self):
        """Set the tick_params of the figure using the method `plt.tick_params`."""
        plt.tick_params(**self.config_dict)


class AbstractTicks(AbstractMatWrap):
    def __init__(
        self, manual_values: typing.Optional[typing.List[float]] = None, **kwargs
    ):
        """
        The settings used to customize a figure's y and x ticks using the `YTicks` and `XTicks` objects.

        This object wraps the following Matplotlib methods:

        - plt.yticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.yticks.html
        - plt.xticks: https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.xticks.html

        Parameters
        ----------
        manual_values : [float]
            Manually override the tick labels to display the labels as the input list of floats.
        """
        super().__init__(**kwargs)

        self.manual_values = manual_values

    def tick_values_from(self, min_value: float, max_value: float) -> np.ndarray:
        """
        Calculate the ticks used for the yticks or xticks from input values of the minimum and maximum coordinate
        values of the y and x axis.

        Parameters
        ----------
        min_value : float
            the minimum value of the ticks that figure is plotted using.
        max_value : float
            the maximum value of the ticks that figure is plotted using.
        """
        if self.manual_values is not None:
            return np.linspace(min_value, max_value, len(self.manual_values))
        else:
            return np.linspace(min_value, max_value, 5)

    def tick_values_in_units_from(
        self, array: array_2d.Array2D, min_value: float, max_value: float, units: Units
    ) -> typing.Optional[np.ndarray]:
        """
        Calculate the labels used for the yticks or xticks from input values of the minimum and maximum coordinate
        values of the y and x axis.

        The values are converted to the `Units` of the figure, via its conversion factor or using data properties.

        Parameters
        ----------
        array : array_2d.Array2D
            The array of data that is to be plotted, whose 2D shape is used to determine the tick values in units of
            pixels if this is the units specified by `units`.
        min_value : float
            the minimum value of the ticks that figure is plotted using.
        max_value : float
            the maximum value of the ticks that figure is plotted using.
        units : Units
            The units the tick values are plotted using.
        """

        if self.manual_values is not None:
            return np.asarray(self.manual_values)
        elif not units.use_scaled:
            return np.linspace(0, array.shape_native[0], 5).astype("int")
        elif (units.use_scaled and units.conversion_factor is None) or not units.in_kpc:
            return np.round(np.linspace(min_value, max_value, 5), 2)
        elif units.use_scaled and units.conversion_factor is not None:
            return np.round(
                np.linspace(
                    min_value * units.conversion_factor,
                    max_value * units.conversion_factor,
                    5,
                ),
                2,
            )

        else:
            raise exc.PlottingException(
                "The tick labels cannot be computed using the input options."
            )


class YTicks(AbstractTicks):
    def set(
        self,
        array: typing.Optional[array_2d.Array2D],
        min_value: float,
        max_value: float,
        units: Units,
    ):
        """
        Set the y ticks of a figure using the shape of an input `Array2D` object and input units.

        Parameters
        -----------
        array : array_2d.Array2D
            The 2D array of data which is plotted.
        min_value : float
            the minimum value of the yticks that figure is plotted using.
        max_value : float
            the maximum value of the yticks that figure is plotted using.
        units : Units
            The units of the figure.
        """

        ticks = self.tick_values_from(min_value=min_value, max_value=max_value)
        labels = self.tick_values_in_units_from(
            array=array, min_value=min_value, max_value=max_value, units=units
        )
        plt.yticks(ticks=ticks, labels=labels, **self.config_dict)


class XTicks(AbstractTicks):
    def set(
        self,
        array: typing.Optional[array_2d.Array2D],
        min_value: float,
        max_value: float,
        units: Units,
    ):
        """
        Set the x ticks of a figure using the shape of an input `Array2D` object and input units.

        Parameters
        -----------
        array : array_2d.Array2D
            The 2D array of data which is plotted.
        min_value : float
            the minimum value of the xticks that figure is plotted using.
        max_value : float
            the maximum value of the xticks that figure is plotted using.
        units : Units
            The units of the figure.
        """

        ticks = self.tick_values_from(min_value=min_value, max_value=max_value)
        labels = self.tick_values_in_units_from(
            array=array, min_value=min_value, max_value=max_value, units=units
        )
        plt.xticks(ticks=ticks, labels=labels, **self.config_dict)


class Title(AbstractMatWrap):
    def __init__(self, **kwargs):
        """
        The settings used to customize the figure's title.

        This object wraps the following Matplotlib methods:

        - plt.title: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.title.html

        The title will automatically be set if not specified, using the name of the function used to plot the data.
        """

        super().__init__(**kwargs)

        self.manual_label = self.kwargs.get("label")

    def set(self, auto_title=None):

        config_dict = self.config_dict

        label = auto_title if self.manual_label is None else self.manual_label

        if "label" in config_dict:
            config_dict.pop("label")

        plt.title(label=label, **config_dict)


class AbstractLabel(AbstractMatWrap):
    def __init__(self, **kwargs):
        """
        The settings used to customize the figure's title and y and x labels.

        This object wraps the following Matplotlib methods:

        - plt.ylabel: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.ylabel.html
        - plt.xlabel: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.xlabel.html

        The y and x labels will automatically be set if not specified, using the input `Unit`'s. object.

        Parameters
        ----------
        units : Units
            The units the data is plotted using.
        manual_label : str
            A manual label which overrides the default computed via the units if input.
        """

        super().__init__(**kwargs)

        self.manual_label = self.kwargs.get("label")

    def label_from_units(self, units: Units) -> typing.Optional[str]:
        """
        Returns the label of an object, by determining it from the figure units if the label is not manually specified.

        Parameters
        ----------
        units : Units
           The units of the data structure that is plotted which informs the appropriate label text.
        """

        if units is None:
            return None

        if units.in_kpc is not None:
            if units.in_kpc:
                return "kpc"
            else:
                return "arcsec"

        if units.use_scaled:
            return "scaled"
        else:
            return "pixels"


class YLabel(AbstractLabel):
    def set(self, units: Units, include_brackets: bool = True, auto_label=None):
        """
        Set the y labels of the figure, including the fontsize.

        The y labels are always the distance scales, thus the labels are either arc-seconds or kpc and depending on
        the unit_label the figure is plotted in.

        Parameters
        -----------
         unit : Units
            The units of the image that is plotted which informs the appropriate y label text.
        include_brackets : bool
            Whether to include brackets around the y label text of the units.
        """

        config_dict = self.config_dict

        if "label" in self.config_dict:
            config_dict.pop("label")

        if self.manual_label is not None:
            plt.ylabel(ylabel=self.manual_label, **config_dict)
        elif auto_label is not None:
            plt.ylabel(ylabel=auto_label, **config_dict)
        else:
            if include_brackets:
                plt.ylabel(
                    ylabel=f"y ({self.label_from_units(units=units)})", **config_dict
                )
            else:
                plt.ylabel(ylabel=self.label_from_units(units=units), **config_dict)


class XLabel(AbstractLabel):
    def set(self, units: Units, include_brackets: bool = True, auto_label=None):
        """
        Set the x labels of the figure, including the fontsize.

        The x labels are always the distance scales, thus the labels are either arc-seconds or kpc and depending on
        the unit_label the figure is plotted in.

        Parameters
        -----------
         unit : Units
            The units of the image that is plotted which informs the appropriate x label text.
        include_brackets : bool
            Whether to include brackets around the x label text of the units.
        """

        config_dict = self.config_dict

        if "label" in self.config_dict:
            config_dict.pop("label")

        if self.manual_label is not None:
            plt.xlabel(xlabel=self.manual_label, **config_dict)
        elif auto_label is not None:
            plt.xlabel(xlabel=auto_label, **config_dict)
        else:
            if include_brackets:
                plt.xlabel(
                    xlabel=f"x ({self.label_from_units(units=units)})", **config_dict
                )
            else:
                plt.xlabel(xlabel=self.label_from_units(units=units), **config_dict)


class Legend(AbstractMatWrap):
    """
    The settings used to include and customize a legend on a figure.

    This object wraps the following Matplotlib methods:

    - plt.legend: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.legend.html

    Parameters
    ----------
    include : bool
        If the legend should be plotted and therefore included on the figure.
    """

    def __init__(self, include=True, **kwargs):

        super().__init__(**kwargs)

        self.include = include

    def set(self):

        if self.include:

            config_dict = self.config_dict
            config_dict.pop("include") if "include" in config_dict else None
            config_dict.pop("include_2d") if "include_2d" in config_dict else None

            plt.legend(**config_dict)


class Output:
    def __init__(
        self,
        path: str = None,
        filename: str = None,
        format: str = None,
        bypass: bool = False,
    ):
        """
        Sets how the figure or subplot is output, either by displaying it on the screen or writing it to hard-disk.

        This object wraps the following Matplotlib methods:

        - plt.show: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.show.html
        - plt.savefig: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.savefig.html

        The default behaviour is the display the figure on the computer screen, as opposed to outputting to hard-disk
        as a file.

        Parameters
        ----------
        path : str
            If the figure is output to hard-disk the path of the folder it is saved to.
        filename : str
            If the figure is output to hard-disk the filename used to save it.
        format : str
            The format of the output, 'show' displays on the computer screen, 'png' outputs to .png, 'fits' outputs to
            `.fits` format.
        bypass : bool
            Whether to bypass the `plt.show` or `plt.savefig` methods, used when plotting a subplot.
        """
        self.path = path

        if path is not None and path:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

        self.filename = filename
        self._format = format
        self.bypass = bypass

    @property
    def format(self) -> str:
        if self._format is None:
            return "show"
        else:
            return self._format

    def to_figure(
        self,
        structure: typing.Optional[abstract_structure.AbstractStructure],
        auto_filename=None,
    ):
        """Output the figure, by either displaying it on the user's screen or to the hard-disk as a .png or .fits file.

        Parameters
        -----------
        structure : abstract_structure.AbstractStructure
            The 2D array of image to be output, required for outputting the image as a fits file.
        """

        filename = auto_filename if self.filename is None else self.filename

        if not self.bypass:
            if self.format == "show":
                plt.show()
            elif self.format == "png":
                plt.savefig(
                    path.join(self.path, f"{filename}.png"), bbox_inches="tight"
                )
            elif self.format == "fits":
                if structure is not None:
                    structure.output_to_fits(
                        file_path=path.join(self.path, f"{filename}.fits"),
                        overwrite=True,
                    )

    def subplot_to_figure(self, auto_filename=None):
        """Output a subhplot figure, either as an image on the screen or to the hard-disk as a .png or .fits file."""

        filename = auto_filename if self.filename is None else self.filename

        if self.format == "show":
            plt.show()
        elif self.format == "png":
            plt.savefig(path.join(self.path, f"{filename}.png"), bbox_inches="tight")


def remove_spaces_and_commas_from_colors(colors):

    colors = [color.strip(",") for color in colors]
    colors = [color.strip(" ") for color in colors]
    colors = list(filter(None, colors))
    if len(colors) == 1:
        return colors[0]
    return colors
