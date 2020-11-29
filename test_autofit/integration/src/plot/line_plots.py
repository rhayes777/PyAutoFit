import matplotlib.pyplot as plt

# The 'line_plots.py' module is unchanged from the previous tutorial.


def line(
    xvalues,
    line,
    ylabel=None,
    output_path=None,
    output_filename=None,
    output_format="show",
):
    """Plot a 1D line of data on a plot of x versus y, where the x-axis is the x coordinate of the line and the y-axis
    is the intensity of the line at that coordinate.

    The function include options to output the image to the hard-disk as a .png.

    Parameters
    ----------
    xvalues : np.ndarray
        The x-coordinates the line profile is defined on.
    line : np.ndarray
        The intensity values of the line profile which are plotted.
    ylabel : str
        The y-label of the plot.
    output_path : str
        The path the image is to be output to hard-disk as a .png.
    output_filename : str
        The filename of the file if it is output as a .png.
    output_format : str
        Determines where the plot is displayed on your screen ("show") or output to the hard-disk as a png ("png").
    """
    plt.plot(xvalues, line)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    if "show" in output_format:
        plt.show()
    elif "png" in output_format:
        plt.savefig(path.join(output_path, f"{output_filename}.png"))
    plt.clf()
