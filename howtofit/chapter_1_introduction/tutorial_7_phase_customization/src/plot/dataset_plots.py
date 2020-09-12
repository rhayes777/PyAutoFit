from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.plot import (
    line_plots,
)

"""The 'dataset_plots.py' module is unchanged from the previous tutorial."""


def data(dataset, output_path=None, output_filename=None, output_format="show"):
    """Plot the data values of a Line _Dataset_.

    Parameters
    -----------
    Line : _Dataset_.Line
        The observed Line _Dataset_ whose data is plotted.
    """
    line_plots.line(
        xvalues=dataset.xvalues,
        line=dataset.data,
        ylabel="Data Values",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def noise_map(dataset, output_path=None, output_filename=None, output_format="show"):
    """Plot the noise-map of a Line _Dataset_.

    Parameters
    -----------
    Line : _Dataset_.Line
        The observed Line _Dataset_ whose data is plotted.
    """
    line_plots.line(
        xvalues=dataset.xvalues,
        line=dataset.noise_map,
        ylabel="Noise-Map",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
