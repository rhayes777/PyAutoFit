import numpy as np
import scipy.optimize

from autofit import exc
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer, persistent_timer
from autofit.optimize.non_linear.non_linear import logger


class DownhillSimplex(NonLinearOptimizer):

    def __init__(self, phase_name, phase_tag=None, phase_folders=None, model_mapper=None, fmin=scipy.optimize.fmin):

        super(DownhillSimplex, self).__init__(phase_name=phase_name, phase_tag=phase_tag, phase_folders=phase_folders,
                                              model_mapper=model_mapper)

        self.xtol = self.config("xtol", float)
        self.ftol = self.config("ftol", float)
        self.maxiter = self.config("maxiter", int)
        self.maxfun = self.config("maxfun", int)

        self.full_output = self.config("full_output", int)
        self.disp = self.config("disp", int)
        self.retall = self.config("retall", int)

        self.fmin = fmin

        logger.debug("Creating DownhillSimplex NLO")

    def copy_with_name_extension(self, extension):
        copy = super().copy_with_name_extension(extension)
        copy.fmin = self.fmin
        copy.xtol = self.xtol
        copy.ftol = self.ftol
        copy.maxiter = self.maxiter
        copy.maxfun = self.maxfun
        copy.full_output = self.full_output
        copy.disp = self.disp
        copy.retall = self.retall
        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(self, nlo, analysis, instance_from_physical_vector, image_path):
            super().__init__(nlo, analysis, image_path)
            self.instance_from_physical_vector = instance_from_physical_vector

        def __call__(self, vector):
            try:
                instance = self.instance_from_physical_vector(vector)
                likelihood = self.fit_instance(instance)
            except exc.FitException:
                likelihood = -np.inf
            return -2 * likelihood

    @persistent_timer
    def fit(self, analysis):
        self.save_model_info()
        initial_vector = self.variable.physical_values_from_prior_medians

        fitness_function = DownhillSimplex.Fitness(self, analysis, self.variable.instance_from_physical_vector,
                                                   self.image_path)

        logger.info("Running DownhillSimplex...")
        output = self.fmin(fitness_function, x0=initial_vector)
        logger.info("DownhillSimplex complete")

        self.backup()
        res = fitness_function.result

        # Create a set of Gaussian priors from this result and associate them with the result object.
        res.gaussian_tuples = [(mean, 0) for mean in output]
        res.previous_variable = self.variable

        analysis.visualize(instance=res.constant, image_path=self.image_path, during_analysis=False)
        return res
