from dynesty import plotting as dyplot
import logging

from autofit.plot import SamplesPlotter
from autofit.plot.samples_plotters import skip_plot_in_test_mode

logger = logging.getLogger(__name__)

class DynestyPlotter(SamplesPlotter):

    @skip_plot_in_test_mode
    def boundplot(self, **kwargs):

        dyplot.boundplot(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="boundplot")
        self.close()

    @skip_plot_in_test_mode
    def cornerbound(self, **kwargs):

        dyplot.cornerbound(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerbound")
        self.close()

    @skip_plot_in_test_mode
    def cornerplot(self, **kwargs):

        try:

            dyplot.cornerplot(
                results=self.samples.results_internal,
                labels=self.model.parameter_labels_with_superscripts_latex,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="cornerplot")
            self.close()

        except ValueError:

            logger.info(
                "Dynesty unable to produce cornerplot visual: posterior estimate therefore"
                "not yet sufficient for this model-fit is not yet robust enough to do this. Visual"
                "should be produced in later update, once posterior estimate is updated."
            )

    @skip_plot_in_test_mode
    def cornerpoints(self, **kwargs):

        try:
            dyplot.cornerpoints(
                results=self.samples.results_internal,
                labels=self.model.parameter_labels_with_superscripts_latex,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="cornerpoints")
            self.close()

        except ValueError:

            logger.info(
                "Dynesty unable to produce cornerpoints visual: posterior estimate therefore"
                "not yet sufficient for this model-fit is not yet robust enough to do this. Visual"
                "should be produced in later update, once posterior estimate is updated."
            )

    @skip_plot_in_test_mode
    def runplot(self, **kwargs):

        try:
            dyplot.runplot(
                results=self.samples.results_internal,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="runplot")
            self.close()

        except ValueError:

            logger.info(
                "Dynesty unable to produce runplot visual: posterior estimate therefore"
                "not yet sufficient for this model-fit is not yet robust enough to do this. Visual"
                "should be produced in later update, once posterior estimate is updated."
            )

    @skip_plot_in_test_mode
    def traceplot(self, **kwargs):

        try:

            dyplot.traceplot(
                results=self.samples.results_internal,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="traceplot")
            self.close()

        except ValueError:

            logger.info(
                "Dynesty unable to produce traceplot visual: posterior estimate therefore"
                "not yet sufficient for this model-fit is not yet robust enough to do this. Visual"
                "should be produced in later update, once posterior estimate is updated."
            )