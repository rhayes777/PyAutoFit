import numpy as np

from autofit import exc
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.log import logger
from autofit.non_linear.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.samples import OptimizerSamples, Sample


class AbstractPySwarms(AbstractOptimizer):
    def __init__(
            self,
            name=None,
            path_prefix=None,
            prior_passer=None,
            initializer=None,
            iterations_per_update=None,
            number_of_cores=None,
            **kwargs
    ):
        """
        A PySwarms Particle Swarm Optimizer global non-linear search.

        For a full description of PySwarms, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/pyswarms
        https://pyswarms.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name : str
            The name of the search, controlling the last folder results are output.
        path_prefix : str
            The path of folders prefixing the name folder where results are output.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        initializer : non_linear.initializer.Initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            **kwargs
        )

        self.number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        logger.debug("Creating PySwarms NLO")

    class Fitness(AbstractOptimizer.Fitness):
        def __call__(self, parameters):

            figures_of_merit = []

            for params_of_particle in parameters:

                try:
                    figure_of_merit = self.figure_of_merit_from_parameters(
                        parameters=params_of_particle
                    )
                except exc.FitException:
                    figure_of_merit = -2.0 * self.resample_figure_of_merit

                figures_of_merit.append(figure_of_merit)

            return np.asarray(figures_of_merit)

        def figure_of_merit_from_parameters(self, parameters):
            """The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. *PySwarms*
            uses the chi-squared value, which is the -2.0*log_posterior."""
            try:
                return -2.0 * self.log_posterior_from_parameters(parameters=parameters)
            except exc.FitException:
                raise exc.FitException

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None):
        """
        Fit a model using PySwarms and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the `NonLinearSearch` seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.

        Returns
        -------
        A result object comprising the Samples object that inclues the maximum log likelihood instance and full
        chains used by the fit.
        """
        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis, pool_ids=pool_ids
        )

        if self.paths.is_object("points"):

            init_pos = self.load_points[-1]
            total_iterations = self.load_total_iterations

            logger.info("Existing PySwarms samples found, resuming non-linear search.")

        else:

            initial_unit_parameters, initial_parameters, initial_log_posteriors = self.initializer.initial_samples_from_model(
                total_points=self.config_dict["n_particles"],
                model=model,
                fitness_function=fitness_function,
            )

            init_pos = np.zeros(shape=(self.config_dict["n_particles"], model.prior_count))

            for index, parameters in enumerate(initial_parameters):

                init_pos[index, :] = np.asarray(parameters)

            total_iterations = 0

            logger.info("No PySwarms samples found, beginning new non-linear search. ")

        lower_bounds = []
        upper_bounds = []

        for key, value in model.prior_class_dict.items():
            lower_bounds.append(key.lower_limit)
            upper_bounds.append(key.upper_limit)

        bounds = (np.asarray(lower_bounds), np.asarray(upper_bounds))

        logger.info("Running PySwarmsGlobal Optimizer...")

        while total_iterations < self.config_dict["iters"]:

            pso = self.sampler_from(
                model=model,
                fitness_function=fitness_function,
                bounds=bounds,
                init_pos=init_pos
            )

            iterations_remaining = self.config_dict["iters"] - total_iterations

            if self.iterations_per_update > iterations_remaining:
                iterations = iterations_remaining
            else:
                iterations = self.iterations_per_update

            if iterations > 0:

                pso.optimize(objective_func=fitness_function.__call__, iters=iterations)

                total_iterations += iterations

                self.paths.save_object(
                    "total_iterations",
                    total_iterations
                )
                self.paths.save_object(
                    "points",
                    pso.pos_history
                )
                self.paths.save_object(
                    "log_posteriors",
                    [-0.5 * cost for cost in pso.cost_history]
                )

                self.perform_update(
                    model=model, analysis=analysis, during_analysis=True
                )

                init_pos = self.load_points[-1]

        logger.info("PySwarmsGlobal complete")

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None, pool_ids=None):

        return PySwarmsGlobal.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_via_sampler_from_model,
            log_likelihood_cap=log_likelihood_cap,
            pool_ids=pool_ids,
        )

    def sampler_from(self, model, fitness_function, bounds, init_pos):
        raise NotImplementedError()

    def samples_via_sampler_from_model(self, model):
        """
        Create an *OptimizerSamples* object from this non-linear search's output files on the hard-disk and model.

        For PySwarms, all quantities are extracted via pickled states of the particle and cost histories.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        parameters = [
            param.tolist() for parameters in self.load_points for param in parameters
        ]
        log_priors = [
            sum(model.log_priors_from_vector(vector=vector)) for vector in parameters
        ]
        log_posteriors = self.load_log_posteriors
        log_likelihoods = [lp - prior for lp, prior in zip(log_posteriors, log_priors)]
        weights = len(log_likelihoods) * [1.0]

        return OptimizerSamples(
            model=model,
            samples=Sample.from_lists(
                parameters=[parameters.tolist()[0] for parameters in self.load_points],
                log_likelihoods=log_likelihoods,
                log_priors=log_priors,
                weights=weights,
                model=model
            ),
            time=self.timer.time
        )

    @property
    def load_total_iterations(self):
        return self.paths.load_object(
            "total_iterations"
        )

    @property
    def load_points(self):
        return self.paths.load_object(
            "points"
        )

    @property
    def load_log_posteriors(self):
        return self.paths.load_object(
            "log_posteriors"
        )


class PySwarmsGlobal(AbstractPySwarms):

    def __init__(
            self,
            name=None,
            path_prefix=None,
            prior_passer=None,
            initializer=None,
            iterations_per_update=None,
            number_of_cores=None,
            **kwargs
    ):
        """
        A PySwarms Particle Swarm Optimizer global non-linear search.

        For a full description of PySwarms, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/pyswarms

        https://pyswarms.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name : str
            The name of the search, controlling the last folder results are output.
        path_prefix : str
            The path of folders prefixing the name folder where results are output.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        initializer : non_linear.initializer.Initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            **kwargs
        )

        logger.debug("Creating PySwarms NLO")

    def sampler_from(self, model, fitness_function, bounds, init_pos):
        """Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""

        import pyswarms

        options = {
            "c1": self.config_dict["cognitive"],
            "c2": self.config_dict["social"],
            "w": self.config_dict["inertia"]
        }

        config_dict = self.config_dict
        config_dict.pop("iters")
        config_dict.pop("cognitive")
        config_dict.pop("social")
        config_dict.pop("inertia")

        return pyswarms.global_best.GlobalBestPSO(
            dimensions=model.prior_count,
            bounds=bounds,
            init_pos=init_pos,
            options=options,
            **config_dict
        )


class PySwarmsLocal(AbstractPySwarms):

    def __init__(
            self,
            name=None,
            path_prefix=None,
            prior_passer=None,
            iterations_per_update=None,
            number_of_cores=None,
            **kwargs
    ):
        """
        A PySwarms Particle Swarm Optimizer global non-linear search.

        For a full description of PySwarms, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/pyswarms

        https://pyswarms.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name : str
            The name of the search, controlling the last folder results are output.
        path_prefix : str
            The path of folders prefixing the name folder where results are output.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        initializer : non_linear.initializer.Initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            prior_passer=prior_passer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            **kwargs
        )

        logger.debug("Creating PySwarms NLO")

    def sampler_from(self, model, fitness_function, bounds, init_pos):
        """
        Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables.
        """

        import pyswarms

        options = {
            "c1": self.config_dict["cognitive"],
            "c2": self.config_dict["social"],
            "w": self.config_dict["inertia"],
            "k": self.config_dict["number_of_k_neighbors"],
            "p": self.config_dict["minkowski_p_norm"],
        }

        config_dict = self.config_dict
        config_dict.pop("iters")
        config_dict.pop("cognitive")
        config_dict.pop("social")
        config_dict.pop("inertia")
        config_dict.pop("number_of_k_neighbors")
        config_dict.pop("minkowski_p_norm")

        return pyswarms.local_best.LocalBestPSO(
            dimensions=model.prior_count,
            bounds=bounds,
            init_pos=init_pos,
            options=options,
            **config_dict
        )
