from howtofit.chapter_1_introduction.tutorial_4_source_code.src.plot import line_plots

"""
These functions are simple matplotlib calls that plot components of our FitDataset class, specifically its
data, noise-map, signal-to-noise-map, residual-map, normalized residual-map and chi-squared-map.

Storing simple functions like this for plotting components of our fit will prove beneficial in later tutorials,
when it comes to inspecting the results of a model-fit after they have been completed.
"""


def data(fit):
    """Plot the data values of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit dataset whose data is plotted.
    """
    line_plots.line(xvalues=fit.xvalues, line=fit.data, ylabel="Data Values")


def noise_map(fit):
    """Plot the noise-map of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit whose noise-map is plotted.
    """
    line_plots.line(xvalues=fit.xvalues, line=fit.noise_map, ylabel="Noise Map")


def signal_to_noise_map(fit):
    """Plot the signal-to-noise-map of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit whose signal-to-noise-map is plotted.
    """
    line_plots.line(
        xvalues=fit.xvalues, line=fit.signal_to_noise_map, ylabel="Signal-To-Noise Map"
    )


def model_data(fit):
    """Plot the model data of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The Fit model data.
    """
    line_plots.line(xvalues=fit.xvalues, line=fit.model_data, ylabel="Model Data")


def residual_map(fit):
    """Plot the residual-map of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit whose residual-map is plotted.
    """
    line_plots.line(xvalues=fit.xvalues, line=fit.residual_map, ylabel="Residual Map")


def normalized_residual_map(fit):
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
    )


def chi_squared_map(fit):
    """Plot the chi_squared-map of a Fit.

    Parameters
    -----------
    Fit : fit.Fit
        The observed Fit whose chi-squared-map is plotted.
    """
    line_plots.line(
        xvalues=fit.xvalues, line=fit.chi_squared_map, ylabel="Chi-Squared Map"
    )
