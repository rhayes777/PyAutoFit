from anesthetic.samples import NestedSamples
from anesthetic import make_2d_axes
from functools import wraps
import numpy as np

from autofit.non_linear.plot import SamplesPlotter
from autofit.non_linear.plot.samples_plotters import skip_plot_in_test_mode


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

        This wrapper is specific to Dynesty, which raises a `ValueError` when visualization is performed before
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
        except (ValueError, KeyError, AssertionError, IndexError, np.linalg.LinAlgError):
            self.log_plot_exception(func.__name__)

    return wrapper


class NestPlotter(SamplesPlotter):

    @skip_plot_in_test_mode
    @log_value_error
    def corner(self, **kwargs):
        """
        Plots the in-built ``dynesty`` plot ``cornerplot``.

        This figure plots a corner plot of the 1-D and 2-D marginalized posteriors.
        """

        samples = NestedSamples(
            np.asarray(self.samples.parameter_lists),
            weights=self.samples.weight_list,
            columns=self.model.parameter_labels_with_superscripts_latex
        )

        # prior = samples.prior()
        fig, axes = make_2d_axes(
            self.model.parameter_labels_with_superscripts_latex,
            figsize=(12, 12),
            facecolor='w'
        )
    #    prior.plot_2d(axes, alpha=0.9, label="prior")
        samples.plot_2d(axes, alpha=0.9, label="posterior")
        axes.iloc[-1, 0].legend(bbox_to_anchor=(len(axes) / 2, len(axes)), loc='lower center', ncols=2)

        self.output.to_figure(auto_filename="corner")
        self.close()
