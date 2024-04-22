from anesthetic.samples import NestedSamples
from anesthetic import make_2d_axes
from functools import wraps
import numpy as np
import warnings

from autoconf import conf

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
        except (ValueError, KeyError, AssertionError, IndexError, TypeError, RuntimeError, np.linalg.LinAlgError):
            pass

    return wrapper


class NestPlotter(SamplesPlotter):

    @skip_plot_in_test_mode
    @log_value_error
    def corner_anesthetic(self, **kwargs):
        """
        Plots a corner plot via the visualization library `anesthetic`.

        This plots a corner plot including the 1-D and 2-D marginalized posteriors.
        """

        config_dict = conf.instance["visualize"]["plots_settings"]["corner_anesthetic"]

        import matplotlib.pylab as pylab

        params = {'font.size' : int(config_dict["fontsize"])}
        pylab.rcParams.update(params)

        figsize = (
            self.model.total_free_parameters * config_dict["figsize_per_parammeter"],
            self.model.total_free_parameters * config_dict["figsize_per_parammeter"]
        )

        samples = NestedSamples(
            np.asarray(self.samples.parameter_lists),
            weights=self.samples.weight_list,
            columns=self.model.parameter_labels_with_superscripts_latex
        )

        from pandas.errors import SettingWithCopyWarning
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

        fig, axes = make_2d_axes(
            self.model.parameter_labels_with_superscripts_latex,
            figsize=figsize,
            facecolor=config_dict["facecolor"],
        )

        warnings.filterwarnings("default", category=SettingWithCopyWarning)

        # prior = samples.prior()
        # prior.plot_2d(axes, alpha=0.9, label="prior")
        samples.plot_2d(
            axes,
            alpha=config_dict["alpha"],
            label="posterior",
        )
        axes.iloc[-1, 0].legend(
            bbox_to_anchor=(len(axes) / 2, len(axes)),
            loc='lower center',
            ncols=2
        )

        self.output.to_figure(auto_filename="corner_anesthetic")
        self.close()
