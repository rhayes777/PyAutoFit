from dynesty import plotting as dyplot

from autofit.plot import SamplesPlotter

from autofit.plot.samples_plotters import skip_plot_in_test_mode
from autofit.plot.samples_plotters import log_value_error

class DynestyPlotter(SamplesPlotter):

    @skip_plot_in_test_mode
    def boundplot(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``boundplot``.
        
        This figure plots the bounding distribution used to propose either (1) live points
        at a given iteration or (2) a specific dead point during
        the course of a run, projected onto the two dimensions specified
        by `dims`.
        """

        dyplot.boundplot(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="boundplot")
        self.close()

    @skip_plot_in_test_mode
    def cornerbound(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``cornerbound``.

        This figure plots the bounding distribution used to propose either (1) live points
        at a given iteration or (2) a specific dead point during
        the course of a run, projected onto all pairs of dimensions.
        """
        
        dyplot.cornerbound(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerbound")
        self.close()

    @skip_plot_in_test_mode
    @log_value_error
    def cornerplot(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``cornerplot``.

        This figure plots a corner plot of the 1-D and 2-D marginalized posteriors.
        """
        dyplot.cornerplot(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")
        self.close()

    @skip_plot_in_test_mode
    @log_value_error
    def cornerpoints(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``cornerpoints``.

        This figure plots a (sub-)corner plot of (weighted) samples.
        """
        dyplot.cornerpoints(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerpoints")
        self.close()

    @skip_plot_in_test_mode
    @log_value_error
    def runplot(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``runplot``.

        This figure plots live points, ln(likelihood), ln(weight), and ln(evidence)
        as a function of ln(prior volume).
        """
        dyplot.runplot(
            results=self.samples.results_internal,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="runplot")
        self.close()

    @skip_plot_in_test_mode
    @log_value_error
    def traceplot(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``traceplot``.

        This figure plots traces and marginalized posteriors for each parameter.
        """
        dyplot.traceplot(
            results=self.samples.results_internal,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="traceplot")
        self.close()