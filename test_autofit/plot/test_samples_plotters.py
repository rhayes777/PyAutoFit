from os import path
import pytest
import autofit.plot as aplt


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "samples"
    )


def test__individual_attributes_are_output(
    samples, plot_path, plot_patch
):

    visuals_1d = aplt.Visuals1D()

    samples_plotter = aplt.SamplesPlotter(
        samples=samples,
        visuals_1d=visuals_1d,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
    )

    samples_plotter.figures_1d(
        progress=True
    )

    assert path.join(plot_path, "progress.png") in plot_patch.paths

    plot_patch.paths = []

    samples_plotter.figures_1d(
        progress=False
    )

    assert path.join(plot_path, "progress.png") not in plot_patch.paths


# def test__subplot_is_output(
#     samples, grid_2d_irregular_7x7_list, mask_2d_7x7, plot_path, plot_patch
# ):
#
#     visuals_1d = aplt.Visuals1D()
#
#     samples_plotter = aplt.SamplesPlotter(
#         samples=samples,
#         visuals_1d=visuals_1d,
#         mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(plot_path, format="png")),
#     )
#
#     samples_plotter.subplot_samples()
#
#     assert path.join(plot_path, "subplot_samples.png") in plot_patch.paths

