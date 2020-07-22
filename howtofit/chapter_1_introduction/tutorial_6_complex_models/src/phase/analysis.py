import autofit as af

from howtofit.chapter_1_introduction.tutorial_6_complex_models.src.fit import fit as f
from howtofit.chapter_1_introduction.tutorial_6_complex_models.src.phase import (
    visualizer,
)

"""
The 'analysis.py' module in this tutorial is changed from tutorial 5, such that the fit function now assumes multiple
profiles are passed to it rather than a single Gaussian, see the comments below!
"""


class Analysis(af.Analysis):
    def __init__(self, masked_dataset, image_path=None):

        super().__init__()

        """The masked dataset and visualizer are created in the same way as tutorial 5."""

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

        """
        In tutorials 3 & 4, the instance was an instance of a single Gaussian profile. PyAutoFit knew this instance
        would contain just one Gaussian, because when the phase was created we used a _PriorModel_ object in PyAutoFit
        to make the Gaussian. This meant we could create the model data using the line:

            model_data = instance.gaussian.profile_from_xvalues(xvalues=self.masked_dataset.xvalues)

        In this tutorial our instance is comprised of multiple Profile objects. This is reflected the main tutorial
        script where we create a model using a CollectionPriorModel:

            model = CollectionPriorModel(gaussian=profiles.Gaussian, exponential=profiles.Exponential).

        By using a CollectionPriorModel, this means the instance parameter input into the fit function is a
        dictionary where individual profiles (and their parameters) can be accessed as followed:
        
            print(instance.profiles.gaussian)
            print(instance.profiles.exponential)
            print(instance.profiles.exponential.centre)

        The names of the attributes of the instance correspond to what we input into the CollectionPriorModel. Lets
        look at a second example:

            model = CollectionPriorModel(
                          gaussian_0=profiles.Gaussian,
                          gaussian_1=profiles.Gaussian,
                          whatever_i_want=profiles.Exponential
                     ).

            print(instance.profiles.gaussian_0)
            print(instance.profiles.gaussian_1)
            print(instance.profiles.whatever_i_want.centre)

        A CollectionPriorModel allows us to name our model components whatever we want!

        In this tutorial, we want our 'fit' function to fit the data with a profile which is the summed profile
        of all individual profiles in the model. Look at 'model_data_from_instance' to see how we do this.
        """

        model_data = self.model_data_from_instance(instance=instance)
        fit = self.fit_from_model_data(model_data=model_data)
        return fit.log_likelihood

    def model_data_from_instance(self, instance):

        """
        To create the summed profile of all individual profiles in an instance, we can use a list comprehension
        to iterate over all profiles in the instance.
        """

        return sum(
            [
                profile.profile_from_xvalues(xvalues=self.masked_dataset.xvalues)
                for profile in instance.profiles
            ]
        )

        """
        For those not familiar with list comprehensions, below I've included how one would use the instance to create
        the summed profile using a more simple for loop.
        
            model_data = np.zeros(shape=self.masked_dataset.xvalues.shape[0])

            for profile in instance:
                model_data += profile.profile_from_xvalues(xvalues=self.masked_dataset.xvalues)
    
            return model_data
        """

    def fit_from_model_data(self, model_data):
        return f.FitDataset(masked_dataset=self.masked_dataset, model_data=model_data)

    def visualize(self, instance, during_analysis):

        """
        It is worth noting here why we create specific methods for creating the model_data and fit in an analysis.
        By doing so, the code in our visualize function (and also in the 'result.py' module) do not need changing
        even though we are now using a model with more components, requiring to sum their individual profiles.
        """

        model_data = self.model_data_from_instance(instance=instance)
        fit = self.fit_from_model_data(model_data=model_data)

        self.visualizer.visualize_fit(fit=fit, during_analysis=during_analysis)
