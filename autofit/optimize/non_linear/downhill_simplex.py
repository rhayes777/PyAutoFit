import numpy as np
import scipy.optimize

from autofit import exc
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import logger
from autofit.optimize.non_linear.samples import AbstractSamples
from autofit.optimize.non_linear.paths import Paths


class DownhillSimplex(NonLinearOptimizer):
    def _fit(self, model, fitness_function):
        raise NotImplementedError()

    def __init__(
            self,
            paths=None,
            fmin=scipy.optimize.fmin
    ):
        if paths is None:
            paths = Paths(non_linear_name=type(self).__name__.lower())
        super().__init__(paths)

        self.xtol = self.config("search", "xtol", float)
        self.ftol = self.config("search", "ftol", float)
        self.maxiter = self.config("search", "maxiter", int)
        self.maxfun = self.config("search", "maxfun", int)

        self.full_output = self.config("search", "full_output", int)
        self.disp = self.config("search", "disp", int)
        self.retall = self.config("search", "retall", int)

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

        def __init__(self, paths, model, analysis, samples_fom_model):

            super().__init__(
                paths=paths,
                analysis=analysis,
                model=model,
                samples_from_model=samples_fom_model
            )

        def __call__(self, vector):
            try:
                instance = self.model.instance_from_vector(vector)
                log_likelihood = self.fit_instance(instance)
            except exc.FitException:
                log_likelihood = -np.inf
            return -2 * log_likelihood

    def _full_fit(self, model, analysis):

        initial_vector = model.physical_values_from_prior_medians

        fitness_function = DownhillSimplex.Fitness(
            paths=self.paths, model=model, analysis=analysis, samples_fom_model=self.samples_from_model
        )

        logger.info("Running DownhillSimplex...")
        samples = self.fmin(fitness_function, x0=initial_vector)
        logger.info("DownhillSimplex complete")

        res = fitness_function.result

        # Create a set of Gaussian priors from this result and associate them with the result object.
        res.gaussian_tuples = [(mean, 0) for mean in samples]
        res.previous_model = model

        analysis.visualize(instance=res.instance, during_analysis=False)
        self.paths.backup_zip_remove()
        return res

    def samples_from_model(self, model):
        pass