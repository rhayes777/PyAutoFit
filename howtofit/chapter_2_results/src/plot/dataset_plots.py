from howtofit.chapter_2_results.src.plot import line_plots

"""The `dataset_plots.py` module is unchanged from the previous tutorial."""


def data(dataset, output_path=None, output_filename=None, output_format="show"):
    """Plot the data values of a Line `Dataset`.

    Parameters
    -----------
    Line : `Dataset`.Line
        The observed Line `Dataset` whose data is plotted.
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
    )


def noise_map(dataset, output_path=None, output_filename=None, output_format="show"):
    """Plot the noise-map of a Line `Dataset`.

    Parameters
    -----------
    Line : `Dataset`.Line
        The observed Line `Dataset` whose data is plotted.
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
    )
