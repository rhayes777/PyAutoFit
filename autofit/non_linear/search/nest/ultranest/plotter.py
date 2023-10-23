from functools import wraps
from autofit.plot import SamplesPlotter

from autofit.plot.samples_plotters import skip_plot_in_test_mode

def log_value_error(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        Prevent an exception terminating the run if visualization fails due to convergence not yet being reached.

        Searches attempt to perform visualization every `iterations_per_update`, however these visualization calls
        may occur before the search has converged on enough of parameter to successfully perform visualization.

        This can lead the search to raise an exception which terminates the Python script, when we instead
        want the code to continue running, to continue the search and perform visualization on a subsequent iteration
        once convergence has been achieved.

        This wrapper catches these exceptions, logs them so the user can see visualization failed and then
        continues the code without raising an exception in a way that terminates the script.

        This wrapper is specific to UltraNest, which raises a `KeyError` when visualization is performed before
        convergence has been achieved.

        Parameters
        ----------
        self
            An instance of a `SearchPlotter` class.
        args
            The arguments used to perform a visualization of the search.
        kwargs
            The keyword arguments used to perform a visualization of the search.
        """
        try:
            return func(self, *args, **kwargs)
        except (KeyError, AssertionError):
            self.log_plot_exception(func.__name__)

    return wrapper

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
            results=self.samples.search_internal,
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
            results=self.samples.search_internal,
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
            results=self.samples.search_internal,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="traceplot")
        self.close()