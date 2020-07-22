import autofit as af

# The 'result.py' module is unchanged from the previous tutorial.


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

        return self.analysis.model_data_from_instance(
            instance=self.samples.max_log_likelihood_instance
        )

    @property
    def max_log_likelihood_fit(self):
        return self.analysis.fit_from_model_data(
            model_data=self.max_log_likelihood_model_data
        )
