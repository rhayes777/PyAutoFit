import autofit as af

from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.fit import (
    fit as f,
)
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.phase import (
    visualizer,
)

"""The 'analysis.py' module in this tutorial is unchanged from the previous tutorial."""


class Analysis(af.Analysis):
    def __init__(self, masked_dataset, image_path=None):

        super().__init__()

        """The masked dataset and visualizer are created in the same way as tutorial 6."""

        self.masked_dataset = masked_dataset

        self.visualizer = visualizer.Visualizer(
            masked_dataset=self.masked_dataset, image_path=image_path
        )

    def log_likelihood_function(self, instance):
        """Determine the fit of a list of Profiles (Gaussians, Exponentials, etc.) to the dataset, using a
        model instance.

        Parameters
        ----------
        instance
            The list of Profile model instance (e.g. the Gaussians, Exponentials, etc.).

        Returns
        -------
        fit : Fit.log_likelihood
            The log likelihood value indicating how well this model fit the masked dataset.
        """
        model_data = self.model_data_from_instance(instance=instance)
        fit = self.fit_from_model_data(model_data=model_data)
        return fit.log_likelihood

    def model_data_from_instance(self, instance):
        return sum(
            [
                profile.profile_from_xvalues(xvalues=self.masked_dataset.xvalues)
                for profile in instance.profiles
            ]
        )

    def fit_from_model_data(self, model_data):
        return f.FitDataset(masked_dataset=self.masked_dataset, model_data=model_data)

    def visualize(self, instance, during_analysis):

        model_data = self.model_data_from_instance(instance=instance)
        fit = self.fit_from_model_data(model_data=model_data)

        self.visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
