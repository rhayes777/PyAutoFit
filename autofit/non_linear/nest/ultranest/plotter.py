from autofit.plot import SamplesPlotter

from autofit.plot.samples_plotters import skip_plot_in_test_mode
from autofit.plot.samples_plotters import log_value_error

class UltraNestPlotter(SamplesPlotter):

    @skip_plot_in_test_mode
    @log_value_error
    def cornerplot(self, **kwargs):
        """
        Plots the in-built ``ultranest`` plot ``cornerplot``.

        This figure plots a corner plot of the 1-D and 2-D marginalized posteriors.
        """

        from ultranest import plot

        plot.cornerplot(
            results=self.samples.results_internal,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")
        self.close()

    @skip_plot_in_test_mode
    @log_value_error
    def runplot(self, **kwargs):
        """
        Plots the in-built ``ultranest`` plot ``runplot``.

        This figure plots live points, ln(likelihood), ln(weight), and ln(evidence) vs. ln(prior volume).
        """
        from ultranest import plot

        plot.runplot(
            results=self.samples.results_internal,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="runplot")
        self.close()

    @skip_plot_in_test_mode
    @log_value_error
    def traceplot(self, **kwargs):
        """
        Plots the in-built ``ultranest`` plot ``traceplot``.

        This figure plots traces and marginalized posteriors for each parameter.
        """
        from ultranest import plot

        plot.traceplot(
            results=self.samples.results_internal,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="traceplot")
        self.close()