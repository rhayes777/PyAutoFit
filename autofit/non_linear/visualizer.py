from os import path

from autoconf import conf
from autofit.plot.mat_wrap.wrap import wrap_base
from autofit.plot import samples_plotters

from autofit.plot.mat_wrap import mat_plot
from autofit.plot.mat_wrap import include

def setting(section, name):
    return conf.instance["visualize"]["plots"][section][name]


def plot_setting(section, name):
    return setting(section, name)


class Visualizer:
    def __init__(self, visualize_path):

        self.visualize_path = visualize_path

        self.include_1d = include.Include1D()
        self.include_2d = include.Include2D()

    def mat_plot_1d_from(self, subfolders, format="png"):
        return mat_plot.MatPlot1D(
            output=wrap_base.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def mat_plot_2d_from(self, subfolders, format="png"):
        return mat_plot.MatPlotCorner(
            output=wrap_base.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def visualize_samples(self, samples):
        def should_plot(name):
            return plot_setting(section="samples", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders="samples")

        samples_plotter = samples_plotters.SamplesPlotter(
            samples=samples, mat_plot_1d=mat_plot_1d, include_1d=self.include_1d
        )

        samples_plotter.figures_1d(
            progress=should_plot("progress"),
        )
