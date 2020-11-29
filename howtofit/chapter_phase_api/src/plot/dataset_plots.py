import matplotlib.pyplot as plt

from src.dataset import dataset as ds
from src.plot import line_plots

"""
These functions are simple matplotlib calls that plot components of our Line class, specifically its data and
noise-map. We additional include a function that plots the dataset on a single subplot.

Storing simple functions like this for plotting components of our `Dataset` will prove beneficial when using the 
`Aggregator`, when it comes to inspecting the results of a model-fit after they have been completed.
"""


def subplot_dataset(
    dataset: ds.Dataset,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
):
    """
    Plot the `Dataset` using a subplot containing both its data and noise-map.

    Parameters
    -----------
    dataset : Dataset
        The observed `Dataset` which is plotted.
    output_path : str
        The path where the image of the data is output, if saved as a `.png`.
    output_filename : str
        The name of the file the image of the data is output too, if saved as a `.png`.
    output_format : str
        Whether the data is output as a `.png` file ("png") or displayed on the screen ("show").
    """

    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    data(
        dataset=dataset,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=True,
    )
    plt.subplot(1, 2, 2)
    noise_map(
        dataset=dataset,
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
    dataset: ds.Dataset,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
    bypass_show: bool = False,
):
    """
    Plot the data values of a `Dataset` object.

    Parameters
    -----------
    dataset : Dataset
        The observed `Dataset` whose data is plotted.
    output_path : str
        The path where the image of the data is output, if saved as a `.png`.
    output_filename : str
        The name of the file the image of the data is output too, if saved as a `.png`.
    output_format : str
        Whether the data is output as a `.png` file ("png") or displayed on the screen ("show").
    bypass_show : bool
        If `True` the show or savefig function is bypassed. This is used when plotting subplots.
    """
    line_plots.line(
        xvalues=dataset.xvalues,
        line=dataset.data,
        errors=dataset.noise_map,
        title="Data",
        ylabel="Data Values",
        color="k",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=bypass_show,
    )


def noise_map(
    dataset: ds.Dataset,
    output_path: str = None,
    output_filename: str = None,
    output_format: str = "show",
    bypass_show: bool = False,
):
    """
    Plot the noise-map of a `Dataset` object.

    Parameters
    -----------
    dataset : Dataset
        The observed `Dataset` whose noise-map is plotted.
    output_path : str
        The path where the image of the noise-map is output, if saved as a `.png`.
    output_filename : str
        The name of the file the image of the noise-map is output too, if saved as a `.png`.
    output_format : str
        Whether the noise-map is output as a `.png` file ("png") or displayed on the screen ("show").
    bypass_show : bool
        If `True` the show or savefig function is bypassed. This is used when plotting subplots.
    """
    line_plots.line(
        xvalues=dataset.xvalues,
        line=dataset.noise_map,
        title="Noise-Map",
        ylabel="Noise-Map",
        color="k",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        bypass_show=bypass_show,
    )
