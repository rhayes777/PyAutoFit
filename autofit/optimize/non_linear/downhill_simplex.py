import numpy as np
import scipy.optimize

from autofit import exc
from autofit.optimize.non_linear.output import AbstractOutput
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer, persistent_timer
from autofit.optimize.non_linear.non_linear import logger


class DownhillSimplex(NonLinearOptimizer):
    def __init__(self, paths, fmin=scipy.optimize.fmin):

        super().__init__(paths)

        self.xtol = self.config("xtol", float)
        self.ftol = self.config("ftol", float)
        self.maxiter = self.config("maxiter", int)
        self.maxfun = self.config("maxfun", int)

        self.full_output = self.config("full_output", int)
        self.disp = self.config("disp", int)
        self.retall = self.config("retall", int)

        self.fmin = fmin

        logger.debug("Creating DownhillSimplex NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
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
        def __init__(self, paths, analysis, instance_from_physical_vector):
            super().__init__(paths, analysis)
            self.instance_from_physical_vector = instance_from_physical_vector

        def __call__(self, vector):
            try:
                instance = self.instance_from_physical_vector(vector)
                likelihood = self.fit_instance(instance)
            except exc.FitException:
                likelihood = -np.inf
            return -2 * likelihood

    @persistent_timer
    def fit(self, analysis, model):
        dhs_output = AbstractOutput(model, self.paths)
        dhs_output.save_model_info()
        initial_vector = model.physical_values_from_prior_medians

        fitness_function = DownhillSimplex.Fitness(
            self.paths, analysis, model.instance_from_physical_vector
        )

        logger.info("Running DownhillSimplex...")
        output = self.fmin(fitness_function, x0=initial_vector)
        logger.info("DownhillSimplex complete")

        res = fitness_function.result

        # Create a set of Gaussian priors from this result and associate them with the result object.
        res.gaussian_tuples = [(mean, 0) for mean in output]
        res.previous_model = model

        analysis.visualize(instance=res.instance, during_analysis=False)
        self.paths.backup_zip_remove()
        return res
