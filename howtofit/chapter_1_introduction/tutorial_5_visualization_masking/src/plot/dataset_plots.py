from howtofit.chapter_1_introduction.tutorial_5_visualization_masking.src.plot import (
    line_plots,
)

"""
To visualize images during a phase, we need to be able to output them to hard-disk as a file (e.g a .png'). The line
plot function and _Dataset_ plot functions below have been extended to provide this functionality.

The benefit of writing our visualization methods in this style, as separate functions in a specific 'plot' module is
now more clear. In 'visualizer.py', this makes it a lot more straight forward to plot each component of the _Dataset_.
However, the real benefit of this style will become fully apparently in tutorial 6.
"""


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
