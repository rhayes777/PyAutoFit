import os
from os import path
import matplotlib.pyplot as plt

"""
This is the `plot_line` function we saw in chapter 1.
"""


def line(
    xvalues,
    line,
    title=None,
    ylabel=None,
    errors=None,
    color="k",
    output_path=None,
    output_filename=None,
    output_format="show",
    bypass_show=False,
):
    """Plot a 1D line of data on a plot of x versus y, where the x-axis is the x coordinate of the line and the y-axis
    is the intensity of the line at that coordinate.

    The function include options to output the image to the hard-disk as a .png.

    Parameters
    ----------
    xvalues : np.ndarray
        The x-coordinates the profile is defined on.
    line : np.ndarray
        The intensity values of the profile which are plotted.
    ylabel : str
        The y-label of the plot.
    output_path : str
        The path the image is to be output to hard-disk as a .png.
    output_filename : str
        The filename of the file if it is output as a .png.
    output_format : str
        Determines where the plot is displayed on your screen ("show") or output to the hard-disk as a png ("png").
    bypass_show : bool
        If `True` the show or savefig function is bypassed. This is used when plotting subplots.
    """

    plt.errorbar(
        x=xvalues, y=line, yerr=errors, color=color, ecolor="k", elinewidth=1, capsize=2
    )
    plt.title(title)
    plt.xlabel("x value of profile")
    plt.ylabel(ylabel)

    if not bypass_show:

        if "show" in output_format:
            plt.show()
        elif "png" in output_format:
            if not path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(path.join(output_path, f"{output_filename}.png"))

        plt.clf()
