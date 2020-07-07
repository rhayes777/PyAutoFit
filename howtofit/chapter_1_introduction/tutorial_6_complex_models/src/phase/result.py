import autofit as af

"""
The 'result.py' module is unchanged from the previous tutorial, although there is a short comment below worth 
reading.
"""


class Result(af.Result):
    def __init__(self, samples, previous_model, search, analysis):
        """
        The results of a non-linear search performed by a phase.

        Parameters
        ----------
        samples : af.Samples
            A class containing the samples of the non-linear search, including methods to get the maximum log
            likelihood model, errors, etc.
        analysis : Analysis
            The Analysis class used by this model-fit to fit the model to the data.
        """
        super().__init__(samples=samples, previous_model=previous_model, search=search)
        self.analysis = analysis

    @property
    def max_log_likelihood_model_data(self):

        """
        It is worth noting why we store the 'Analysis' class in the Result class. In this tutorial, we changed our
        model and how it created the model-data (e.g. as a sum of profiles). However, we did not need to change
        the result module in any way, because it uses the 'analysis.py' module.

        Had this function explicitly written out how the most likely model-data is created it would of needed to be
        updated, creating more work for ourselves!
        """

        return self.analysis.model_data_from_instance(
            instance=self.samples.max_log_likelihood_instance
        )

    @property
    def max_log_likelihood_fit(self):
        return self.analysis.fit_from_model_data(
            model_data=self.max_log_likelihood_model_data
        )
