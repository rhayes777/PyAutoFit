from dynesty import plotting as dyplot
import logging

from autofit.plot import SamplesPlotter
from autofit.plot.samples_plotters import skip_plot_in_test_mode

logger = logging.getLogger(__name__)

class DynestyPlotter(SamplesPlotter):
    
    @staticmethod
    def log_plot_exception(plot_name : str):
        """
        Plotting the results of a ``dynesty`` model-fit before they have converged on an
        accurate estimate of the posterior can lead the ``dynesty`` plotting routines
        to raise a ``ValueError``.

        This exception is caught in each of the plotting methods below, and this
        function is used to log the behaviour.

        Parameters
        ----------
        plot_name
            The name of the ``dynesty`` plot which raised a ``ValueError``
        """

        logger.info(
            f"Dynesty unable to produce {plot_name} visual: posterior estimate therefore"
            "not yet sufficient for this model-fit is not yet robust enough to do this. Visual"
            "should be produced in later update, once posterior estimate is updated."
        )

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
    def cornerplot(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``cornerplot``.

        This figure plots a corner plot of the 1-D and 2-D marginalized posteriors.
        """
        try:

            dyplot.cornerplot(
                results=self.samples.results_internal,
                labels=self.model.parameter_labels_with_superscripts_latex,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="cornerplot")
            self.close()

        except ValueError:

            self.log_plot_exception(plot_name="cornerplot")

    @skip_plot_in_test_mode
    def cornerpoints(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``cornerpoints``.

        This figure plots a (sub-)corner plot of (weighted) samples.
        """
        try:
            dyplot.cornerpoints(
                results=self.samples.results_internal,
                labels=self.model.parameter_labels_with_superscripts_latex,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="cornerpoints")
            self.close()

        except ValueError:

            self.log_plot_exception(plot_name="cornerpoints")

    @skip_plot_in_test_mode
    def runplot(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``runplot``.

        This figure plots live points, ln(likelihood), ln(weight), and ln(evidence)
        as a function of ln(prior volume).
        """
        try:
            dyplot.runplot(
                results=self.samples.results_internal,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="runplot")
            self.close()

        except ValueError:

            self.log_plot_exception(plot_name="runplot")

    @skip_plot_in_test_mode
    def traceplot(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``traceplot``.

        This figure plots traces and marginalized posteriors for each parameter.
        """
        try:

            dyplot.traceplot(
                results=self.samples.results_internal,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="traceplot")
            self.close()

        except ValueError:

            self.log_plot_exception(plot_name="traceplot")