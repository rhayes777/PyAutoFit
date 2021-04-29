from autoarray.plot.mat_wrap.wrap import wrap_base

wrap_base.set_backend()

import matplotlib.pyplot as plt
from matplotlib import patches as ptch
from matplotlib.collections import PatchCollection
import numpy as np
import itertools
import typing
from typing import List, Union

from autoarray.inversion import mappers
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.structures.vector_fields import vector_field_irregular
from autoarray import exc


class AbstractMatWrap2D(wrap_base.AbstractMatWrap):
    """
    An abstract base class for wrapping matplotlib plotting methods which take as input and plot data structures. For
    example, the `ArrayOverlay` object specifically plots `Array2D` data structures.

    As full description of the matplotlib wrapping can be found in `mat_base.AbstractMatWrap`.
    """

    @property
    def config_folder(self):
        return "mat_wrap_2d"


class ArrayOverlay(AbstractMatWrap2D):
    """
    Overlays an `Array2D` data structure over a figure.

    This object wraps the following Matplotlib method:

    - plt.imshow: https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.imshow.html

    This uses the `Units` and coordinate system of the `Array2D` to overlay it on on the coordinate system of the
    figure that is plotted.
    """

    def overlay_array(self, array, figure):

        aspect = figure.aspect_from_shape_native(shape_native=array.shape_native)
        extent = array.extent_of_zoomed_array(buffer=0)

        plt.imshow(X=array.native, aspect=aspect, extent=extent, **self.config_dict)


class GridScatter(AbstractMatWrap2D):
    """
    Scatters an input set of grid points, for example (y,x) coordinates or data structures representing 2D (y,x)
    coordinates like a `Grid2D` or `Grid2DIrregular`. List of (y,x) coordinates are plotted with varying colors.

    This object wraps the following Matplotlib methods:

    - plt.scatter: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html

    There are a number of children of this method in the `mat_obj.py` module that plot specific sets of (y,x)
    points. Each of these objects uses uses their own config file and settings so that each has a unique appearance
    on every figure:

    - `OriginScatter`: plots the (y,x) coordinates of the origin of a data structure (e.g. as a black cross).
    - `MaskScatter`: plots a mask over an image, using the `Mask2d` object's (y,x)  `edge_grid_sub_1` property.
    - `BorderScatter: plots a border over an image, using the `Mask2d` object's (y,x) `border_grid_sub_1` property.
    - `PositionsScatter`: plots the (y,x) coordinates that are input in a plotter via the `positions` input.
    - `IndexScatter`: plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.
    - `PixelizationGridScatter`: plots the grid of a `Pixelization` object (see `autoarray.inversion`).

    Parameters
    ----------
    colors : [str]
        The color or list of colors that the grid is plotted using. For plotting indexes or a grid list, a
        list of colors can be specified which the plot cycles through.
    """

    def scatter_grid(self, grid: typing.Union[np.ndarray, grid_2d.Grid2D]):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        """

        config_dict = self.config_dict

        if len(config_dict["c"]) > 1:
            config_dict["c"] = config_dict["c"][0]

        try:
            plt.scatter(y=grid[:, 0], x=grid[:, 1], **config_dict)
        except (IndexError, TypeError):
            return self.scatter_grid_list(grid_list=grid)

    def scatter_grid_list(
        self,
        grid_list: Union[List[grid_2d.Grid2D], List[grid_2d_irregular.Grid2DIrregular]],
    ):
        """
         Plot an input list of grids of (y,x) coordinates using the matplotlib method `plt.scatter`.

         This method colors each grid in each entry of the list the same, so that the different grids are visible in
         the plot.

         Parameters
         ----------
         grid_list : [Grid2DIrregular]
             The list of grids of (y,x) coordinates that are plotted.
         """
        if len(grid_list) == 0:
            return

        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        try:
            for grid in grid_list:

                plt.scatter(y=grid[:, 0], x=grid[:, 1], c=next(color), **config_dict)
        except IndexError:
            return None

    def scatter_grid_colored(
        self,
        grid: typing.Union[np.ndarray, grid_2d.Grid2D],
        color_array: np.ndarray,
        cmap: str,
    ):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        The method colors the scattered grid according to an input ndarray of color values, using an input colormap.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        color_array : ndarray
            The array of RGB color values used to color the grid.
        cmap : str
            The Matplotlib colormap used for the grid point coloring.
        """

        config_dict = self.config_dict
        config_dict.pop("c")

        plt.scatter(y=grid[:, 0], x=grid[:, 1], c=color_array, cmap=cmap, **config_dict)

    def scatter_grid_indexes(
        self, grid: typing.Union[np.ndarray, grid_2d.Grid2D], indexes: np.ndarray
    ):
        """
        Plot specific points of an input grid of (y,x) coordinates, which are specified according to the 1D or 2D
        indexes of the `Grid2D`.

        This method allows us to color in points on grids that map between one another.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        indexes : np.ndarray
            The 1D indexes of the grid that are colored in when plotted.
        """
        if not isinstance(grid, np.ndarray):
            raise exc.PlottingException(
                "The grid passed into scatter_grid_indexes is not a ndarray and thus its"
                "1D indexes cannot be marked and plotted."
            )

        if len(grid.shape) != 2:
            raise exc.PlottingException(
                "The grid passed into scatter_grid_indexes is not 2D (e.g. a flattened 1D"
                "grid) and thus its 1D indexes cannot be marked."
            )

        if isinstance(indexes, list):
            if not any(isinstance(i, list) for i in indexes):
                indexes = [indexes]

        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        for index_list in indexes:

            if all([isinstance(index, float) for index in index_list]) or all(
                [isinstance(index, int) for index in index_list]
            ):

                plt.scatter(
                    y=grid[index_list, 0],
                    x=grid[index_list, 1],
                    color=next(color),
                    **config_dict,
                )

            elif all([isinstance(index, tuple) for index in index_list]) or all(
                [isinstance(index, list) for index in index_list]
            ):

                ys, xs = map(list, zip(*index_list))

                plt.scatter(
                    y=grid.native[ys, xs, 0],
                    x=grid.native[ys, xs, 1],
                    color=next(color),
                    **config_dict,
                )

            else:

                raise exc.PlottingException(
                    "The indexes input into the grid_scatter_index method do not conform to a "
                    "useable type"
                )


class GridPlot(AbstractMatWrap2D):
    """
    Plots `Grid2D` data structure that are better visualized as solid lines, for example rectangular lines that are
    plotted over an image and grids of (y,x) coordinates as lines (as opposed to a scatter of points
    using the `GridScatter` object).

    This object wraps the following Matplotlib methods:

    - plt.plot: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html

    Parameters
    ----------
    colors : [str]
        The color or list of colors that the grid is plotted using. For plotting indexes or a grid list, a
        list of colors can be specified which the plot cycles through.
    """

    def plot_rectangular_grid_lines(
        self,
        extent: typing.Tuple[float, float, float, float],
        shape_native: typing.Tuple[int, int],
    ):
        """
        Plots a rectangular grid of lines on a plot, using the coordinate system of the figure.

        The size and shape of the grid is specified by the `extent` and `shape_native` properties of a data structure
        which will provide the rectangaular grid lines on a suitable coordinate system for the plot.

        Parameters
        ----------
        extent : (float, float, float, float)
            The extent of the rectangualr grid, with format [xmin, xmax, ymin, ymax]
        shape_native : (int, int)
            The 2D shape of the mask the array is paired with.
        """

        ys = np.linspace(extent[2], extent[3], shape_native[1] + 1)
        xs = np.linspace(extent[0], extent[1], shape_native[0] + 1)

        # grid lines
        for x in xs:
            plt.plot([x, x], [ys[0], ys[-1]], **self.config_dict)
        for y in ys:
            plt.plot([xs[0], xs[-1]], [y, y], **self.config_dict)

    def plot_grid(self, grid: typing.Union[np.ndarray, grid_2d.Grid2D]):
        """
        Plot an input grid of (y,x) coordinates using the matplotlib method `plt.scatter`.

        Parameters
        ----------
        grid : Grid2D
            The grid of (y,x) coordinates that is plotted.
        """
        try:
            plt.plot(grid[:, 1], grid[:, 0], **self.config_dict)
        except (IndexError, TypeError):
            return self.plot_grid_list(grid_list=grid)

    def plot_grid_list(
        self,
        grid_list: Union[List[grid_2d.Grid2D], List[grid_2d_irregular.Grid2DIrregular]],
    ):
        """
         Plot an input list of grids of (y,x) coordinates using the matplotlib method `plt.line`.

        This method colors each grid in the list the same, so that the different grids are visible in the plot.

         This provides an alternative to `GridScatter.scatter_grid_list` where the plotted grids appear as lines
         instead of scattered points.

         Parameters
         ----------
         grid_list : Grid2DIrregular
             The list of grids of (y,x) coordinates that are plotted.
         """

        if len(grid_list) == 0:
            return None

        color = itertools.cycle(self.config_dict["c"])
        config_dict = self.config_dict
        config_dict.pop("c")

        try:
            for grid in grid_list:

                plt.plot(grid[:, 1], grid[:, 0], c=next(color), **config_dict)
        except IndexError:
            return None


class VectorFieldQuiver(AbstractMatWrap2D):
    """
    Plots a `VectorField` data structure. A vector field is a set of 2D vectors on a grid of 2d (y,x) coordinates.
    These are plotted as arrows representing the (y,x) components of each vector at each (y,x) coordinate of it
    grid.

    This object wraps the following Matplotlib method:

    https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.quiver.html
    """

    def quiver_vector_field(
        self, vector_field: vector_field_irregular.VectorField2DIrregular
    ):
        """
         Plot a vector field using the matplotlib method `plt.quiver` such that each vector appears as an arrow whose
         direction depends on the y and x magnitudes of the vector.

         Parameters
         ----------
         vector_field : VectorField2DIrregular
             The vector field that is plotted using `plt.quiver`.
         """
        plt.quiver(
            vector_field.grid[:, 1],
            vector_field.grid[:, 0],
            vector_field[:, 1],
            vector_field[:, 0],
            **self.config_dict,
        )


class PatchOverlay(AbstractMatWrap2D):
    """
    Adds patches to a plotted figure using matplotlib `patches` objects.

    The coordinate system of each `Patch` uses that of the figure, which is typically set up using the plotted
    data structure. This makes it straight forward to add patches in specific locations.

    This object wraps methods described in below:

    https://matplotlib.org/3.3.2/api/collections_api.html
    """

    def overlay_patches(self, patches: typing.Union[ptch.Patch]):
        """
        Overlay a list of patches on a figure, for example an `Ellipse`.
        `
        Parameters
        ----------
        patches : [Patch]
            The patches that are laid over the figure.
        """

        patch_collection = PatchCollection(patches=patches, **self.config_dict)

        plt.gcf().gca().add_collection(patch_collection)


class VoronoiDrawer(AbstractMatWrap2D):
    """
    Draws Voronoi pixels from a `MapperVoronoi` object (see `inversions.mapper`). This includes both drawing
    each Voronoi cell and coloring it according to a color value.

    The mapper contains the grid of (y,x) coordinate where the centre of each Voronoi cell is plotted.

    This object wraps methods described in below:

    https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.fill.html
    """

    def draw_voronoi_pixels(
        self,
        mapper: mappers.MapperVoronoi,
        values: np.ndarray,
        cmap: wrap_base.Cmap,
        colorbar: wrap_base.Colorbar,
        colorbar_tickparams: wrap_base.ColorbarTickParams = None,
    ):
        """
        Draws the Voronoi pixels of the input `mapper` using its `pixelization_grid` which contains the (y,x) 
        coordinate of the centre of every Voronoi cell. This uses the method `plt.fill`.
        
        Parameters
        ----------
        mapper : MapperVoronoi
            An object which contains the (y,x) grid of Voronoi cell centres.
        values : np.ndarray
            An array used to compute the color values that every Voronoi cell is plotted using.
        cmap : str
            The colormap used to plot each Voronoi cell.
        colorbar : Colorbar
            The `Colorbar` object in `mat_base` used to set the colorbar of the figure the Voronoi mesh is plotted on.
        """
        regions, vertices = self.voronoi_polygons(voronoi=mapper.voronoi)

        if values is not None:

            vmin = cmap.vmin_from_array(array=values)
            vmax = cmap.vmax_from_array(array=values)

            color_values = np.where(values > vmax, vmax, values)
            color_values = np.where(values < vmin, vmin, color_values)

            if vmax != vmin:
                color_array = (color_values - vmin) / (vmax - vmin)
            else:
                color_array = np.ones(color_values.shape[0])

            cmap = plt.get_cmap(cmap.config_dict["cmap"])
            colorbar = colorbar.set_with_color_values(
                cmap=cmap, color_values=color_values
            )
            if colorbar is not None and colorbar_tickparams is not None:
                colorbar_tickparams.set(cb=colorbar)
        else:
            cmap = plt.get_cmap("Greys")
            color_array = np.zeros(shape=mapper.pixels)

        for region, index in zip(regions, range(mapper.pixels)):
            polygon = vertices[region]
            col = cmap(color_array[index])
            plt.fill(*zip(*polygon), facecolor=col, zorder=-1, **self.config_dict)

    def voronoi_polygons(self, voronoi, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite regions.

        Parameters
        ----------
        voronoi : Voronoi
            The input Voronoi diagram that is being plotted.
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Grid2DIrregular for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """

        if voronoi.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = voronoi.vertices.tolist()

        center = voronoi.points.mean(axis=0)
        if radius is None:
            radius = voronoi.points.ptp().max() * 2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(voronoi.ridge_points, voronoi.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(voronoi.point_region):
            vertices = voronoi.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = voronoi.points[p2] - voronoi.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # hyper

                midpoint = voronoi.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = voronoi.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)


