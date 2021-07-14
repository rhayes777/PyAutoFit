from typing import Optional

from sqlalchemy.orm import Session

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.samples import OptimizerSamples, Sample
from autofit.non_linear.abstract_search import Analysis
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.initializer import Initializer

import copy
from scipy import optimize
import numpy as np

class LBFGS(AbstractOptimizer):
    __identifier_fields__ = ()

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            prior_passer: Optional[PriorPasser] = None,
            initializer: Optional[Initializer] = None,
            iterations_per_update: int = None,
            session: Optional[Session] = None,
            **kwargs
    ):
        """
        A L-BFGS scipy non-linear search.

        For a full description of the scipy L-BFGS method, checkout its documentation:

        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

        If you use `LBFGS` as part of a published work, please cite the package via scipy following the instructions
        under the *Attribution* section of the GitHub page.

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        prior_passer
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores: int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            session=session,
            **kwargs
        )

        self.logger.debug("Creating LBFGS Search")

    @property
    def config_dict_options(self):

        config_dict = copy.copy(self._class_config["options"]._dict)

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

        return config_dict

    def _fit(
            self,
            model: AbstractPriorModel,
            analysis: Analysis,
            log_likelihood_cap: Optional[float] = None
    ):
        """
        Fit a model using the scipy L-BFGS method and the Analysis class which contains the data and returns the log
        likelihood from instances of the model, which the `NonLinearSearch` seeks to maximize.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data,
            returning the log likelihood the `NonLinearSearch` maximizes.

        Returns
        -------
        A result object comprising the Samples object that inclues the maximum log likelihood instance and full
        chains used by the fit.
        """
        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis
        )

        if self.paths.is_object("x0"):

            x0 = self.paths.load_object("x0")
            total_iterations = self.paths.load_object("total_iterations")

            self.logger.info("Existing LBGFS samples found, resuming non-linear search.")

        else:

            unit_parameter_lists, parameter_lists, log_posterior_list = self.initializer.samples_from_model(
                total_points=1,
                model=model,
                fitness_function=fitness_function,
            )

            x0 = np.asarray(parameter_lists[0])

            total_iterations = 0

            self.logger.info("No LBFGS samples found, beginning new non-linear search. ")

        maxiter = self.config_dict_options.get("maxiter", 1e8)

        while total_iterations < maxiter:

            iterations_remaining = maxiter - total_iterations

            iterations = min(self.iterations_per_update, iterations_remaining)

            if iterations > 0:

                config_dict_options = self.config_dict_options
                config_dict_options["maxiter"] = iterations

                lbfgs = optimize.minimize(
                    fun=fitness_function.__call__,
                    x0=x0,
                    method="L-BFGS-B",
                    options=config_dict_options,
                    **self.config_dict_search
                )

                total_iterations += lbfgs.nit

                self.paths.save_object(
                    "total_iterations",
                    total_iterations
                )
                self.paths.save_object(
                    "log_posterior",
                    fitness_function.log_posterior_from(parameter_list=lbfgs.x)
                )
                self.paths.save_object(
                    "x0",
                    lbfgs.x
                )

                self.perform_update(
                    model=model, analysis=analysis, during_analysis=True
                )

                x0 = lbfgs.x

                if lbfgs.nit < iterations:
                    return

        self.logger.info("L-BFGS sampling complete.")

    def samples_from(
            self,
            model: AbstractPriorModel
    ):

        return LBFGSSamples(
            model=model,
            x0=self.paths.load_object("x0"),
            log_posterior_list=np.array([self.paths.load_object("log_posterior")]),
            total_iterations=self.paths.load_object("total_iterations"),
            time=self.timer.time
        )


class LBFGSSamples(OptimizerSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            x0: np.ndarray,
            log_posterior_list: np.ndarray,
            total_iterations: int,
            time: Optional[float] = None,
    ):
        """
        Create an *OptimizerSamples* object from this non-linear search's output files on the hard-disk and model.

        For LBFGS, all quantities are extracted via pickled states of the particle and cost histories.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        self.x0 = x0
        self._log_posterior_list = log_posterior_list
        self.total_iterations = total_iterations

        parameter_lists = [list(self.x0)]
        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameter_lists
        ]
        log_likelihood_list = [lp - prior for lp, prior in zip(self._log_posterior_list, log_prior_list)]
        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        super().__init__(
            model=model,
            sample_list=sample_list,
            time=time,
        )
