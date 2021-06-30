from typing import Optional

from sqlalchemy.orm import Session

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.mle.abstract_mle import AbstractMLE
from autofit.non_linear.abstract_search import Analysis

from scipy import optimize
import numpy as np

class LBFGS(AbstractMLE):
    __identifier_fields__ = ()

    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag: Optional[str] = None,
            prior_passer=None,
            initializer=None,
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
        number_of_cores : int
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

    def _fit(self, model: AbstractPriorModel, analysis : Analysis, log_likelihood_cap : float=None):
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

        optimize.minimize(fun=fitness_function.__call__, method="L-BFGS-B")

        if self.paths.is_object("points"):

            init_pos = self.load_points[-1]
            total_iterations = self.load_total_iterations

            self.logger.info("Existing PySwarms samples found, resuming non-linear search.")

        else:

            initial_unit_parameter_lists, initial_parameter_lists, initial_log_posterior_list = self.initializer.initial_samples_from_model(
                total_points=self.config_dict_search["n_particles"],
                model=model,
                fitness_function=fitness_function,
            )

            init_pos = np.zeros(shape=(self.config_dict_search["n_particles"], model.prior_count))

            for index, parameters in enumerate(initial_parameter_lists):

                init_pos[index, :] = np.asarray(parameters)

            total_iterations = 0

            self.logger.info("No PySwarms samples found, beginning new non-linear search. ")

        while iterations_remaining > 0:

            if self.iterations_per_update > iterations_remaining:
                iterations = iterations_remaining
            else:
                iterations = self.iterations_per_update

            for sample in emcee_sampler.sample(
                    initial_state=emcee_state,
                    iterations=iterations,
                    progress=True,
                    skip_initial_state_check=True,
                    store=True,
            ):
                pass

            emcee_state = emcee_sampler.get_last_sample()

            total_iterations += iterations
            iterations_remaining = self.config_dict_run["nsteps"] - total_iterations

            samples = self.perform_update(
                model=model, analysis=analysis, during_analysis=True
            )

            if self.auto_correlations_settings.check_for_convergence:
                if emcee_sampler.iteration > self.auto_correlations_settings.check_size:
                    if samples.converged:
                        iterations_remaining = 0

        self.logger.info("Emcee sampling complete.")

