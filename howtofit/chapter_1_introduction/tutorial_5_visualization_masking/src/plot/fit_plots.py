from howtofit.chapter_1_introduction.tutorial_5_visualization_masking.src.plot import (
    line_plots,
)

"""
To visualize images during a phase, we need to be able to output them to hard-disk as a file (e.g a .png'). The line
plot function and fit plot functions below have been extended to provide this functionality.

The benefit of writing our visualization methods in this style, as separate functions in a specific 'plot' module is
now more clear. In 'visualizer.py', this makes it a lot more straight forward to plot each component of the fit.
However, the real benefit of this style will become fully apparently in tutorial 6.
"""


def data(fit, output_path=None, output_filename=None, output_format="show"):
    """Plot the data values of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit dataset whose data is plotted.
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.data,
        ylabel="Data Values",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def noise_map(fit, output_path=None, output_filename=None, output_format="show"):
    """Plot the noise-map of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit whose noise-map is plotted.
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.noise_map,
        ylabel="Noise Map",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def signal_to_noise_map(
    fit, output_path=None, output_filename=None, output_format="show"
):
    """Plot the signal-to-noise-map of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit whose signal-to-noise-map is plotted.
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.signal_to_noise_map,
        ylabel="Signal-To-Noise Map",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def model_data(fit, output_path=None, output_filename=None, output_format="show"):
    """Plot the model data of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The Fit model data.
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.model_data,
        ylabel="Model Data",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def residual_map(fit, output_path=None, output_filename=None, output_format="show"):
    """Plot the residual-map of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit whose residual-map is plotted.
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.residual_map,
        ylabel="Residual Map",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def normalized_residual_map(
    fit, output_path=None, output_filename=None, output_format="show"
):
    """Plot the normalized_residual-map of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit whose normalized residual-map is plotted.
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.normalized_residual_map,
        ylabel="Normalized Residual Map",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def chi_squared_map(fit, output_path=None, output_filename=None, output_format="show"):
    """Plot the chi_squared-map of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit whose chi-squared-map is plotted.
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.chi_squared_map,
        ylabel="Chi-Squared Map",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
