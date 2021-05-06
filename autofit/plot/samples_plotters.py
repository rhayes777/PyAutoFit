from autofit.plot.mat_wrap import visuals as vis
from autofit.plot.mat_wrap import include as inc
from autofit.plot.mat_wrap import mat_plot as mp
from autofit.plot import abstract_plotters


class SamplesPlotter(abstract_plotters.AbstractPlotter):
    def __init__(
            self, 
            samples,
            mat_plot_1d: mp.MatPlot1D = mp.MatPlot1D(),
            visuals_1d: vis.Visuals1D = vis.Visuals1D(),
            include_1d: inc.Include1D = inc.Include1D(),
            mat_plot_corner: mp.MatPlotCorner = mp.MatPlotCorner(),
    ):

        self.samples = samples

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
        )

        self.mat_plot_corner = mat_plot_corner

    @property
    def visuals_with_include_2d(self):

        return self.visuals_2d + self.visuals_2d.__class__()

    def figures_1d(
        self,
        progress=False,
    ):
        """Plot each attribute of the imaging data_type as individual figures one by one (e.g. the dataset, noise_map, PSF, \
         Signal-to_noise-map, etc).

        Set *autolens.data_type.array.mat_plot_corner.mat_plot_corner* for a description of all innput parameters not described below.

        Parameters
        -----------
        imaging : data_type.ImagingData
            The imaging data_type, which includes the observed data_type, noise_map, PSF, signal-to-noise_map, etc.
        include_origin : True
            If true, the include_origin of the dataset's coordinate system is plotted as a 'x'.
        """

        if progress:

            self.mat_plot_1d.plot_yx(
                y=self.samples.log_likelihoods,
                x=range(len(self.samples.log_likelihoods)),
                visuals_1d=self.visuals_1d,
                auto_labels=mp.AutoLabels(
                    title=f"Log Likelihood Progress (Max = {max(self.samples.log_likelihoods)}",
                    filename="progress",
                    ylabel="Log Likelihood",
                    xlabel="Likelihood Evaluation Number"
                ),
                plot_axis_type_override="symlog",
            )

    def figure_corner(self, triangle=False):

        if triangle:

            self.mat_plot_corner.plot_corner(samples=self.samples)

    def subplot(
        self,
        progress=False,
        auto_filename="subplot_samples",
    ):

        self._subplot_custom_plot(
            progress=progress,
            auto_labels=mp.AutoLabels(filename=auto_filename),
        )

    def subplot_samples(self):
        self.subplot(
            progress=True
        )

