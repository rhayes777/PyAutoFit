from dynesty import plotting as dyplot
from functools import wraps
import os

from autoconf import conf
from autofit.plot import SamplesPlotter



def skip_plot_in_test_mode(func):
    """
    Skips visualization plots of non-linear searches if test mode is on.

    Parameters
    ----------
    func
        A function which plots a result of a non-linear search.

    Returns
    -------
        A function that plots a visual, or None if test mode is on.
    """

    @wraps(func)
    def wrapper(
        obj: object,
        *args,
        **kwargs
    ):
        """
        Skips visualization plots of non-linear searches if test mode is on.

        Parameters
        ----------
        obj
            An plotter object which performs visualization of a non-linear search.

        Returns
        -------
            A function that plots a visual, or None if test mode is on.
        """

        if "PYAUTOFIT_TEST_MODE" in os.environ:
            return

        return func(obj, *args, **kwargs)

    return wrapper


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

        dyplot.cornerplot(
            results=self.samples.results_internal,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="cornerplot")
        self.close()

    @skip_plot_in_test_mode
    def cornerpoints(self, **kwargs):

        try:
            dyplot.cornerpoints(
                results=self.samples.results_internal,
                labels=self.model.parameter_labels_with_superscripts_latex,
                **kwargs
            )

            self.output.to_figure(structure=None, auto_filename="cornerpoints")
        except ValueError:
            pass

        self.close()

    @skip_plot_in_test_mode
    def runplot(self, **kwargs):

        try:
            dyplot.runplot(
                results=self.samples.results_internal,
                **kwargs
            )
        except ValueError:
            pass

        self.output.to_figure(structure=None, auto_filename="runplot")
        self.close()

    @skip_plot_in_test_mode
    def traceplot(self, **kwargs):

        dyplot.traceplot(
            results=self.samples.results_internal,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="traceplot")
        self.close()