import autofit as af
from src.dataset import dataset as ds
from src.fit import fit as f
from src.phase import visualizer
import numpy as np
import pickle

"""
The `analysis.py` module contains the Analysis class we introduced in chapter 1. Whereas before this class took the
data and noise-map separately, we now pass it an instance of the `Dataset` class. We have also restructured the code
performing the model-fit, so that we can reuse it in the `result.py` module.
"""


class Analysis(af.Analysis):
    def __init__(self, dataset: ds.Dataset, settings):
        """The `Dataset` is created in `phase.py`."""

        self.dataset = dataset
        self.settings = settings

        self.visualizer = visualizer.Visualizer()

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Returns the fit of a list of Profiles (Gaussians, Exponentials, etc.) to the dataset, using a
        model instance.

        Parameters
        ----------
        instance : af.ModelInstance
            The list of Profile model instance (e.g. the Gaussians, Exponentials, etc.).

        Returns
        -------
        fit : Fit.log_likelihood
            The log likelihood value indicating how well this model fit the `MaskedDataset`.
        """

        """
        In chapter 1, the `instance` that came into the `log_likelihood_function` when using multiple profiles was a 
        `CollectionPriorModel`. We accessed the attributes of this instance as follows:
        
            print(instance.gaussian.sigma)
            model_data = instance.gaussian.profile_from_xvalues(xvalues=self.dataset.xvalues)

        In `phase.py` this instance is now set up as a `phase_property`, as seen by the following line of Python code:
         
        profiles = af.PhaseProperty("profiles")
         
        This means the `instance` input into the `log_likelihood_function` has an additional dictionary containing the
        profiles:

            print(instance.profiles.gaussian.sigma)
            model_data = instance.profiles.gaussian.profile_from_xvalues(xvalues=self.dataset.xvalues)

        The names of the attributes of the `profiles` dictionary again correspond to the inputa of the 
        `CollectionPriorModel`. Lets look at a second example:

            model = CollectionPriorModel(
                          gaussian_0=profiles.Gaussian,
                          gaussian_1=profiles.Gaussian,
                          whatever_i_want=profiles.Exponential
                     ).

            print(instance.profiles.gaussian_0)
            print(instance.profiles.gaussian_1)
            print(instance.profiles.whatever_i_want.centre)

        In this example project, we only have one phase property, `profiles`, making this additional dictionary appear
        superflous. However, one can imagine that for complex model-fitting problems we might have many phase 
        properties, and this ability to separate them will make for cleaner and more manageable source code!
        """

        model_data = self.model_data_from_instance(instance=instance)
        fit = self.fit_from_model_data(model_data=model_data)
        return fit.log_likelihood

    def model_data_from_instance(self, instance: af.ModelInstance) -> np.ndarray:
        """
        To create the summed profile of all individual profiles in an instance, we can use a list comprehension
        to iterate over all profiles in the instance.

        Note how we now use `instance.profiles` to get this dictionary, where in chapter ` we simply used `instance`.

        Parameters
        ----------
        instance : af.ModelInstance
            The list of Profile model instance (e.g. the Gaussians, Exponentials, etc.).
        """

        return sum(
            [
                profile.profile_from_xvalues(xvalues=self.dataset.xvalues)
                for profile in instance.profiles
            ]
        )

    def fit_from_model_data(self, model_data: np.ndarray) -> f.FitDataset:
        """
        Call the `FitDataset` class in `fit.py` to create an instance of the fit, whose `log_likelihood` property
        is used in the `log_likelihood_function`.

        Parameters
        ----------
        model_data : np.ndarray
            The model data of the 1D profile(s) the data is fitted with.
        """

        return f.FitDataset(dataset=self.dataset, model_data=model_data)

    def visualize(
        self, paths: af.Paths, instance: af.ModelInstance, during_analysis: bool
    ):

        """
        This visualize function is used in the same fashion as it was in chapter 1. The `Visualizer` class is described
        in tutorial 2 of this chapter.
        """

        model_data = self.model_data_from_instance(instance=instance)
        fit = self.fit_from_model_data(model_data=model_data)

        self.visualizer.visualize_dataset(paths=paths, dataset=self.dataset)
        self.visualizer.visualize_fit(
            paths=paths, fit=fit, during_analysis=during_analysis
        )

    def save_attributes_for_aggregator(self, paths):
        """Save files like the dataset, mask and settings as pickle files so they can be loaded in the ``Aggregator``"""

        # These functions save the objects we will later access using the aggregator. They are saved via the `pickle`
        # module in Python, which serializes the data on to the hard-disk.

        with open(f"{paths.pickle_path}/dataset.pickle", "wb") as f:
            pickle.dump(self.dataset, f)

        with open(f"{paths.pickle_path}/settings.pickle", "wb+") as f:
            pickle.dump(self.settings, f)
