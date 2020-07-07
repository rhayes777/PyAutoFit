import autofit as af

from howtofit.chapter_1_introduction.tutorial_5_visualization_masking.src.fit import (
    fit as f,
)
from howtofit.chapter_1_introduction.tutorial_5_visualization_masking.src.phase import (
    visualizer,
)

"""
The 'analysis.py' module in this tutorial has two changes from the previous tutorial:

   - The Analysis class is passed a mask, which is used to mask the data and thus fit only specific regions of it.
   - The Analysis class has a Visualizer, which performs visualization during and after the non-linear search.
"""


class Analysis(af.Analysis):
    def __init__(self, masked_dataset, image_path=None):

        super().__init__()

        """The masked-dataset is created in the 'phase.py' module, in the 'make_analysis' method."""

        self.masked_dataset = masked_dataset

        """
        The visualizer is the tool that we'll use the visualize a phase's unmasked dataset (before the model-fitting
        begins) and the best-fit solution found by the model-fit (during and after the model-fitting).

        Check out 'visualizer.py' for more details.
        """

        self.visualizer = visualizer.Visualizer(
            masked_dataset=self.masked_dataset, image_path=image_path
        )

    def log_likelihood_function(self, instance):
        """Determine the log likelihood of a fit of a Gaussian to the dataset, using the model instance of a Gaussian.

        Parameters
        ----------
        instance
            The Gaussian model instance.

        Returns
        -------
        fit : Fit.log_likelihood
            The log likelihood value indicating how well this model fit the masked dataset.
        """
        model_data = self.model_data_from_instance(instance=instance)
        fit = self.fit_from_model_data(model_data=model_data)
        return fit.log_likelihood

    def model_data_from_instance(self, instance):
        return instance.gaussian.profile_from_xvalues(
            xvalues=self.masked_dataset.xvalues
        )

    def fit_from_model_data(self, model_data):
        return f.FitDataset(masked_dataset=self.masked_dataset, model_data=model_data)

    def visualize(self, instance, during_analysis):
        """
        During a phase, the 'visualize' method is called throughout the non-linar search. The 'instance' passed into
        the visualize method is highest log likelihood solution obtained by the model-fit so far.

        In the analysis we use this instance to create the best-fit fit of our model-fit.
        """

        model_data = self.model_data_from_instance(instance=instance)
        fit = self.fit_from_model_data(model_data=model_data)

        """The visualizer now outputs images of the best-fit results to hard-disk (checkout 'visualizer.py')."""

        self.visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
