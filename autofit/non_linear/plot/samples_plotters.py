import matplotlib.pyplot as plt
import numpy as np
from functools import wraps
import logging
import os

from autofit.non_linear.plot.output import Output

logger = logging.getLogger(__name__)

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

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        return func(obj, *args, **kwargs)

    return wrapper



class SamplesPlotter:
    def __init__(
            self, 
            samples,
            output : Output = Output()
    ):

        self.samples = samples
        self.output = output

    @property
    def model(self):
        return self.samples.model

    def close(self):
        if plt.fignum_exists(num=1):
            plt.close()

    def log_plot_exception(self, plot_name : str):
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
            f"{self.__class__.__name__} unable to produce {plot_name} visual: posterior estimate therefore"
            "not yet sufficient for this model-fit is not yet robust enough to do this. Visual"
            "should be produced in later update, once posterior estimate is updated."
        )

    def corner(self, **kwargs):
        """
        Plots the `nautilus` plot `corner`, using the package `corner.py`.

        This figure plots a corner plot of the 1-D and 2-D marginalized posteriors.
        """

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        import corner

        corner.corner(
            data=np.asarray(self.samples.parameter_lists),
            weight_list=self.samples.weight_list,
            labels=self.model.parameter_labels_with_superscripts_latex,
            **kwargs
        )

        self.output.to_figure(structure=None, auto_filename="corner")
        self.close()



