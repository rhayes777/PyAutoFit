from autoconf import conf
import typing


class AbstractInclude:
    def __init__(
        self,
    ):
        """
        Sets which `Visuals` are included on a figure that is plotted using a `Plotter`.

        The `Include` object is used to extract the visuals of the plotted data structure (e.g. `Array2D`, `Grid2D`) so
        they can be used in plot functions. Only visuals with a `True` entry in the `Include` object are extracted and t
        plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin : bool
            If `True`, the `origin` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        mask : bool
            if `True`, the `mask` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        """

        pass

    def load(self, value, name):
        if value is True:
            return True
        elif value is False:
            return False
        elif value is None:
            return conf.instance["visualize"]["include"][self.section][name]

    @property
    def section(self):
        raise NotImplementedError


class Include1D(AbstractInclude):
    def __init__(
        self,
    ):
        """
        Sets which `Visuals1D` are included on a figure plotting 1D data that is plotted using a `Plotter1D`.

        The `Include` object is used to extract the visuals of the plotted 1D data structures so they can be used in 
        plot functions. Only visuals with a `True` entry in the `Include` object are extracted and plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin : bool
            If `True`, the `origin` of the plotted data structure (e.g. `Line`)  is included on the figure.
        mask : bool
            if `True`, the `mask` of the plotted data structure (e.g. `Line`)  is included on the figure.
        """
        super().__init__()

    @property
    def section(self):
        return "include_1d"


class Include2D(AbstractInclude):
    def __init__(
        self,
    ):
        """
        Sets which `Visuals2D` are included on a figure plotting 2D data that is plotted using a `Plotter2D`.

        The `Include` object is used to extract the visuals of the plotted 2D data structures so they can be used in 
        plot functions. Only visuals with a `True` entry in the `Include` object are extracted and plotted.

        If an entry is not input into the class (e.g. it retains its default entry of `None`) then the bool is
        loaded from the `config/visualize/include.ini` config file. This means the default visuals of a project
        can be specified in a config file.

        Parameters
        ----------
        origin : bool
            If `True`, the `origin` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        mask : bool
            if `True`, the `mask` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        border : bool
            If `True`, the `border` of the plotted data structure (e.g. `Array2D`, `Grid2D`)  is included on the figure.
        mapper_data_pixelization_grid : bool
            If `True`, the pixelization grid in the data plane of a plotted `Mapper` is included on the figure.
        mapper_source_pixelization_grid : bool
            If `True`, the pixelization grid in the source plane of a plotted `Mapper` is included on the figure.
        parallel_overscan : bool
            If `True`, the parallel overscan of a plotted `Frame2D` is included on the figure.
        serial_prescan : bool
            If `True`, the serial prescan of a plotted `Frame2D` is included on the figure.
        serial_overscan : bool
            If `True`, the serial overscan of a plotted `Frame2D` is included on the figure.
        """

        super().__init__()

    @property
    def section(self):
        return "include_2d"
