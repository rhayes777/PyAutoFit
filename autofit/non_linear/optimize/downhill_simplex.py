import numpy as np
import scipy.optimize

from autofit import exc
from autofit.non_linear.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.paths import Paths

from autofit.non_linear.log import logger

class DownhillSimplex(AbstractOptimizer):
    def __init__(self, paths=None, fmin=scipy.optimize.fmin):

        if paths is None:
            paths = Paths()

        super().__init__(paths)

        self.xtol = self._config("search", "xtol", float)
        self.ftol = self._config("search", "ftol", float)
        self.maxiter = self._config("search", "maxiter", int)
        self.maxfun = self._config("search", "maxfun", int)

        self.full_output = self._config("search", "full_output", int)
        self.disp = self._config("search", "disp", int)
        self.retall = self._config("search", "retall", int)

        self.fmin = fmin

        logger.debug("Creating DownhillSimplex NLO")

    @property
    def tag(self):
        return "dhs"

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

    class Fitness(AbstractOptimizer.Fitness):
        def __init__(self, paths, model, analysis, samples_fom_model):

            super().__init__(
                paths=paths,
                analysis=analysis,
                model=model,
                samples_from_model=samples_fom_model,
            )

        def __call__(self, vector):
            try:
                instance = self.model.instance_from_vector(vector)
                log_likelihood = self.fit_instance(instance)
            except exc.FitException:
                log_likelihood = -np.inf
            return -2 * log_likelihood

    def _fit(self, model, analysis):

        initial_vector = model.physical_values_from_prior_medians

        fitness_function = DownhillSimplex.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_fom_model=self.samples_from_model,
        )

        logger.info("Running DownhillSimplex...")
        samples = self.fmin(fitness_function, x0=initial_vector)
        logger.info("DownhillSimplex complete")

        res = fitness_function.result

        # Create a set of Gaussian priors from this result and associate them with the result object.
        res.gaussian_tuples = [(mean, 0) for mean in samples]
        res.model = model

        analysis.visualize(instance=res.instance, during_analysis=False)
        self.paths.backup_zip_remove()
        return res

    def samples_from_model(self, model):
        pass
