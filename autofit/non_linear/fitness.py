
class Fitness:
    def __init__(
            self, model, analysis, log_likelihood_cap=None
    ):
        

        self.analysis = analysis
        self.model = model
        self.log_likelihood_cap = log_likelihood_cap

    def __call__(self, parameters, *kwargs):
        try:
            figure_of_merit = self.figure_of_merit_from(parameter_list=parameters)

            if np.isnan(figure_of_merit):
                return self.resample_figure_of_merit

            return figure_of_merit

        except exc.FitException:
            return self.resample_figure_of_merit

    def fit_instance(self, instance):
        log_likelihood = self.analysis.log_likelihood_function(instance=instance)

        if self.log_likelihood_cap is not None:
            if log_likelihood > self.log_likelihood_cap:
                log_likelihood = self.log_likelihood_cap

        return log_likelihood

    def log_likelihood_from(self, parameter_list):
        instance = self.model.instance_from_vector(vector=parameter_list)
        log_likelihood = self.fit_instance(instance)

        return log_likelihood

    def log_posterior_from(self, parameter_list):
        log_likelihood = self.log_likelihood_from(parameter_list=parameter_list)
        log_prior_list = self.model.log_prior_list_from_vector(
            vector=parameter_list
        )

        return log_likelihood + sum(log_prior_list)

    def figure_of_merit_from(self, parameter_list):
        """
        The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. This varies
        between different `NonLinearSearch`s, for example:

            - The *Optimizer* *PySwarms* uses the chi-squared value, which is the -2.0*log_posterior.
            - The *MCMC* algorithm *Emcee* uses the log posterior.
            - Nested samplers such as *Dynesty* use the log likelihood.
        """
        raise NotImplementedError()

    @staticmethod
    def prior(cube, model):
        # NEVER EVER REFACTOR THIS LINE! Haha.

        phys_cube = model.vector_from_unit_vector(unit_vector=cube)

        for i in range(len(phys_cube)):
            cube[i] = phys_cube[i]

        return cube

    @staticmethod
    def fitness(cube, model, fitness):
        return fitness(instance=model.instance_from_vector(cube))

    @property
    def resample_figure_of_merit(self):
        """
        If a sample raises a FitException, this value is returned to signify that the point requires resampling or
         should be given a likelihood so low that it is discard.
        """
        return -np.inf