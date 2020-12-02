import matplotlib.pyplot as plt

from src.fit import fit as f
from src.plot import line_plots

"""
These functions are simple matplotlib calls that plot components of our Line class, specifically its data and
noise-map. We additional include a function that plots the dataset on a single subplot.

Storing simple functions like this for plotting components of our `Fit` will prove beneficial when using the 
`Aggregator`, when it comes to inspecting the results of a model-fit after they have been completed.
"""


def subplot_fit(
    fit: f.FitDataset,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
):
    """
    Plot the `FitDataset` using a subplot containing its data, noise-map, model-data, residual map, normalized
    residual map and chi-squared map.

    Parameters
    -----------
    Fit : fit.Fit
        The `FitDataset` whose data and fit quantities are plotted.
    output_path : str
        The path where the fit is output, if saved as a `.png`.
    output_filename : str
        The name of the file the fit is output too, if saved as a `.png`.
    output_format : str
        Whether the data is output as a `.png` file ("png") or displayed on the screen ("show").
    """

    plt.figure(figsize=(18, 12))
    plt.subplot(2, 3, 1)
    data(
        fit=fit,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=True,
    )
    plt.subplot(2, 3, 2)
    noise_map(
        fit=fit,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=True,
    )
    plt.subplot(2, 3, 3)
    model_data(
        fit=fit,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=True,
    )
    plt.subplot(2, 3, 4)
    residual_map(
        fit=fit,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=True,
    )
    plt.subplot(2, 3, 5)
    normalized_residual_map(
        fit=fit,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=True,
    )
    plt.subplot(2, 3, 6)
    chi_squared_map(
        fit=fit,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=True,
    )
    if "show" in output_format:
        plt.show()
    elif "png" in output_format:
        plt.savefig(path.join(output_path, f"{output_filename}.png"))
    plt.clf()


def data(
    fit,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
    bypass_show: bool = False,
):
    """
    Plot the data values of a `FitDataset` object.

    Parameters
    -----------
    Fit : fit.Fit
        The `FitDataset` whose data is plotted.
    output_path : str
        The path where the fit is output, if saved as a `.png`.
    output_filename : str
        The name of the file the fit is output too, if saved as a `.png`.
    output_format : str
        Whether the data is output as a `.png` file ("png") or displayed on the screen ("show").
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.data,
        title="Data",
        ylabel="Data Values",
        color="k",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=bypass_show,
    )


def noise_map(
    fit,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
    bypass_show: bool = False,
):
    """
    Plot the noise-map of a `FitDataset` object.

    Parameters
    -----------
    Fit : fit.Fit
        The `FitDataset` whose noise-map is plotted.
    output_path : str
        The path where the fit is output, if saved as a `.png`.
    output_filename : str
        The name of the file the fit is output too, if saved as a `.png`.
    output_format : str
        Whether the data is output as a `.png` file ("png") or displayed on the screen ("show").
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.noise_map,
        title="Noise-Map",
        ylabel="Noise Map",
        color="k",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=bypass_show,
    )


def model_data(
    fit,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
    bypass_show: bool = False,
):
    """
    Plot the model data of a `FitDataset` object.

    Parameters
    -----------
    Fit : fit.Fit
        The `FitDataset` whose model data.
    output_path : str
        The path where the fit is output, if saved as a `.png`.
    output_filename : str
        The name of the file the fit is output too, if saved as a `.png`.
    output_format : str
        Whether the data is output as a `.png` file ("png") or displayed on the screen ("show").
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.model_data,
        title="Model Data",
        ylabel="Model Data",
        color="r",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=bypass_show,
    )


def residual_map(
    fit,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
    bypass_show: bool = False,
):
    """
    Plot the residual-map of a `FitDataset` object.

    Parameters
    -----------
    Fit : fit.Fit
        The `FitDataset` whose residual-map is plotted.
    output_path : str
        The path where the fit is output, if saved as a `.png`.
    output_filename : str
        The name of the file the fit is output too, if saved as a `.png`.
    output_format : str
        Whether the data is output as a `.png` file ("png") or displayed on the screen ("show").
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.residual_map,
        title="Residual-Map",
        ylabel="Residual-Map",
        color="r",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=bypass_show,
    )


def normalized_residual_map(
    fit,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
    bypass_show: bool = False,
):
    """
    Plot the normalized_residual-map of a `FitDataset` object.

    Parameters
    -----------
    Fit : fit.Fit
        The `FitDataset` whose normalized residual-map is plotted.
    output_path : str
        The path where the fit is output, if saved as a `.png`.
    output_filename : str
        The name of the file the fit is output too, if saved as a `.png`.
    output_format : str
        Whether the data is output as a `.png` file ("png") or displayed on the screen ("show").
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.normalized_residual_map,
        title="Normalized Residual-Map",
        ylabel="Normalized Residual-Map",
        color="r",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=bypass_show,
    )


def chi_squared_map(
    fit,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
    bypass_show: bool = False,
):
    """
    Plot the chi_squared-map of a `FitDataset` object.

    Parameters
    -----------
    Fit : fit.Fit
        The `FitDataset` whose chi-squared-map is plotted.
    output_path : str
        The path where the fit is output, if saved as a `.png`.
    output_filename : str
        The name of the file the fit is output too, if saved as a `.png`.
    output_format : str
        Whether the data is output as a `.png` file ("png") or displayed on the screen ("show").
    """
    line_plots.line(
        xvalues=fit.xvalues,
        line=fit.chi_squared_map,
        title="Chi-Squared Map",
        ylabel="Chi-Squared Map",
        color="r",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=bypass_show,
    )
