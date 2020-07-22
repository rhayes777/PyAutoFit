import matplotlib.pyplot as plt

"""This function plots a line of 1D data, and was already introduce in tutorial 2."""


def line(xvalues, line, ylabel=None):
    """Plot a 1D line of data on a plot of x versus y, where the x-axis is the x coordinate of the line and the y-axis
    is the intensity of the line at that coordinate.

    Parameters
    ----------
    xvalues : ndarray
        The x-coordinates the profile is defined on.
    line : ndarray
        The intensity values of the profile which are plotted.
    ylabel : str
        The y-label of the plot.
    """

    plt.plot(xvalues, line)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()
