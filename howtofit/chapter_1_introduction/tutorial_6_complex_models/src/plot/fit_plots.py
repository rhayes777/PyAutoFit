from howtofit.chapter_1_introduction.tutorial_6_complex_models.src.plot import (
    line_plots,
)

"""This module is unchanged from the previous tutorial."""


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
