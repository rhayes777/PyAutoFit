import os
import numpy as np
import pickle

from autofit import exc
from autofit.non_linear.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.samples import OptimizerSamples

from autofit.non_linear.log import logger


class AbstractPySwarms(AbstractOptimizer):
    def __init__(
        self,
        paths=None,
        prior_passer=None,
        n_particles=None,
        iters=None,
        cognitive=None,
        social=None,
        inertia=None,
        ftol=None,
        initializer=None,
        iterations_per_update=None,
        number_of_cores=None,
    ):
        """
        A PySwarms Particle Swarm Optimizer global non-linear search.

        For a full description of PySwarms, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/pyswarms
        https://pyswarms.readthedocs.io/en/latest/index.html

        A Global-best Particle Swarm Optimization (gbest PSO) algorithm.

        It takes a set of candidate solutions, and tries to find the best solution using a position-velocity update
        method. Uses a star-topology where each particle is attracted to the best performing particle.

        The position update can be defined as:
        xi(t+1)=xi(t)+vi(t+1)

        Where the position at the current timestep t is updated using the computed velocity at t+1.

        Furthermore, the velocity update is defined as:

        vij(t+1)=w∗vij(t)+c1r1j(t)[yij(t)−xij(t)]+c2r2j(t)[y^j(t)−xij(t)]

        Here, c1 and c2 are the cognitive and social parameters respectively. They control the particle’s behavior
        given two choices: (1) to follow its personal best or (2) follow the swarm’s global best position. Overall,
        this dictates if the swarm is explorative or exploitative in nature. In addition, a parameter w controls
        the inertia of the swarm’s movement.

        Extensions:

        Allows runs to be terminated and resumed from the point it was terminated. This is achieved by outputting
        the necessary results (e.g. the points of the particles) during the model-fit after an input number of
        iterations.

        Different options for particle intialization, with the default 'prior' method starting all particles over
        the priors defined by each parameter.

        If you use *PySwarms* as part of a published work, please cite the package following the instructions under the
        *Attribution* section of the GitHub page.

        All remaining attributes are emcee parameters and described at the PySwarms API webpage:

        https://pyswarms.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this non-linear search to a subsequent non-linear search.
        n_particles : int
            The number of particles in the swarm used to sample parameter space.
        iters : int
            The number of iterations that are used to sample parameter space.
        cognitive : float
            The cognitive parameter defining how the PSO particles interact with one another to sample parameter space.
        social : float
            The social parameter defining how the PSO particles interact with one another to sample parameter space.
        inertia : float
            The inertia parameter defining how the PSO particles interact with one another to sample parameter space.
        ftol : float
            Relative error in objective_func(best_pos) acceptable for convergence.
        initializer : non_linear.initializer.Initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        self.n_particles = (
            self._config("search", "n_particles", int)
            if n_particles is None
            else n_particles
        )
        self.iters = self._config("search", "iters", int) if iters is None else iters

        self.cognitive = (
            self._config("search", "cognitive", float)
            if cognitive is None
            else cognitive
        )
        self.social = (
            self._config("search", "social", float) if social is None else social
        )
        self.inertia = (
            self._config("search", "inertia", float) if inertia is None else inertia
        )
        self.ftol = self._config("search", "ftol", float) if ftol is None else ftol

        super().__init__(
            paths=paths,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
        )

        self.number_of_cores = (
            self._config("parallel", "number_of_cores", int)
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
            """The figure of merit is the value that the non-linear search uses to sample parameter space. *PySwarms*
            uses the chi-squared value, which is the -2.0*log_posterior."""
            try:
                return -2.0 * self.log_posterior_from_parameters(parameters=parameters)
            except exc.FitException:
                raise exc.FitException

    def _fit(self, model, analysis):
        """
        Fit a model using PySwarms and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the non-linear search seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the non-linear search maximizes.

        Returns
        -------
        A result object comprising the Samples object that inclues the maximum log likelihood instance and full
        chains used by the fit.
        """
        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis, pool_ids=pool_ids
        )

        if os.path.exists("{}/{}.pickle".format(self.paths.samples_path, "points")):

            init_pos = self.load_points[-1]
            total_iterations = self.load_total_iterations

            logger.info("Existing PySwarms samples found, resuming non-linear search.")

        else:

            initial_unit_parameters, initial_parameters, initial_log_posteriors = self.initializer.initial_samples_from_model(
                total_points=self.n_particles,
                model=model,
                fitness_function=fitness_function,
            )

            init_pos = np.zeros(shape=(self.n_particles, model.prior_count))

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

        while total_iterations < self.iters:

            pso = self.sampler_fom_model_and_fitness(
                model=model,
                fitness_function=fitness_function,
                bounds=bounds,
                init_pos=init_pos,
            )

            iterations_remaining = self.iters - total_iterations

            if self.iterations_per_update > iterations_remaining:
                iterations = iterations_remaining
            else:
                iterations = self.iterations_per_update

            if iterations > 0:

                pso.optimize(objective_func=fitness_function.__call__, iters=iterations)

                total_iterations += iterations

                with open(
                    f"{self.paths.samples_path}/total_iterations.pickle", "wb"
                ) as f:
                    pickle.dump(total_iterations, f)

                with open(f"{self.paths.samples_path}/points.pickle", "wb") as f:
                    pickle.dump(pso.pos_history, f)

                with open(
                    f"{self.paths.samples_path}/log_posteriors.pickle", "wb"
                ) as f:
                    pickle.dump([-0.5 * cost for cost in pso.cost_history], f)

                self.perform_update(
                    model=model, analysis=analysis, during_analysis=True
                )

                init_pos = self.load_points[-1]

        logger.info("PySwarmsGlobal complete")

    @property
    def tag(self):
        """Tag the output folder of the PySwarms non-linear search, according to the number of particles and
        parameters defining the search strategy."""

        name_tag = self._config("tag", "name", str)
        n_particles_tag = f"{self._config('tag', 'n_particles')}_{self.n_particles}"
        cognitive_tag = f"{self._config('tag', 'cognitive')}_{self.cognitive}"
        social_tag = f"{self._config('tag', 'social')}_{self.social}"
        inertia_tag = f"{self._config('tag', 'inertia')}_{self.inertia}"

        return (
            f"{name_tag}__{n_particles_tag}_{cognitive_tag}_{social_tag}_{inertia_tag}"
        )

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the emcee non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.prior_passer = self.prior_passer
        copy.n_particles = self.n_particles
        copy.iters = self.iters
        copy.cognitive = self.cognitive
        copy.social = self.social
        copy.intertia = self.inertia
        copy.ftol = self.ftol
        copy.initializer = self.initializer
        copy.iterations_per_update = self.iterations_per_update
        copy.number_of_cores = self.number_of_cores

        return copy

    def fitness_function_from_model_and_analysis(self, model, analysis, pool_ids=None):

        return PySwarmsGlobal.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from_model,
            pool_ids=pool_ids,
        )

    def sampler_fom_model_and_fitness(self, model, fitness_function):
        raise NotImplementedError()

    def samples_from_model(self, model):
        """Create an *OptimizerSamples* object from this non-linear search's output files on the hard-disk and model.

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
        weights = len(log_likelihoods)*[1.0]

        return OptimizerSamples(
            model=model,
            parameters=[parameters.tolist()[0] for parameters in self.load_points],
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            time=self.timer.time
        )

    @property
    def load_total_iterations(self):
        with open(
            "{}/{}.pickle".format(self.paths.samples_path, "total_iterations"), "rb"
        ) as f:
            return pickle.load(f)

    @property
    def load_points(self):
        with open("{}/{}.pickle".format(self.paths.samples_path, "points"), "rb") as f:
            return pickle.load(f)

    @property
    def load_log_posteriors(self):
        with open(
            "{}/{}.pickle".format(self.paths.samples_path, "log_posteriors"), "rb"
        ) as f:
            return pickle.load(f)


class PySwarmsGlobal(AbstractPySwarms):
    def __init__(
        self,
        paths=None,
        prior_passer=None,
        n_particles=None,
        iters=None,
        cognitive=None,
        social=None,
        inertia=None,
        ftol=None,
        initializer=None,
        iterations_per_update=None,
        number_of_cores=None,
    ):
        """ A PySwarms Particle Swarm Optimizer global non-linear search.

        For a full description of PySwarms, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/pyswarms

        https://pyswarms.readthedocs.io/en/latest/index.html

        A Global-best Particle Swarm Optimization (gbest PSO) algorithm.

        It takes a set of candidate solutions, and tries to find the best solution using a position-velocity update
        method. Uses a star-topology where each particle is attracted to the best performing particle.

        The position update can be defined as:
        xi(t+1)=xi(t)+vi(t+1)

        Where the position at the current timestep t is updated using the computed velocity at t+1.

        Furthermore, the velocity update is defined as:

        vij(t+1)=w∗vij(t)+c1r1j(t)[yij(t)−xij(t)]+c2r2j(t)[y^j(t)−xij(t)]

        Here, c1 and c2 are the cognitive and social parameters respectively. They control the particle’s behavior
        given two choices: (1) to follow its personal best or (2) follow the swarm’s global best position. Overall,
        this dictates if the swarm is explorative or exploitative in nature. In addition, a parameter w controls
        the inertia of the swarm’s movement.

        Extensions:

        - Allows runs to be terminated and resumed from the point it was terminated. This is achieved by outputting the
          necessary results (e.g. the points of the particles) during the model-fit after an input number of iterations.

        - Different options for particle intialization, with the default 'prior' method starting all particles over the
        priors defined by each parameter.

        If you use *PySwarms* as part of a published work, please cite the package following the instructions under the
        *Attribution* section of the GitHub page.

        Parameters
        ----------
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this non-linear search to a subsequent non-linear search.
        n_particles : int
            The number of particles in the swarm used to sample parameter space.
        iters : int
            The number of iterations that are used to sample parameter space.
        cognitive : float
            The cognitive parameter defining how the PSO particles interact with one another to sample parameter space.
        social : float
            The social parameter defining how the PSO particles interact with one another to sample parameter space.
        inertia : float
            The inertia parameter defining how the PSO particles interact with one another to sample parameter space.
        ftol : float
            Relative error in objective_func(best_pos) acceptable for convergence.
        initializer : non_linear.initializer.Initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.

        All remaining attributes are emcee parameters and described at the PySwarms API webpage:

        https://pyswarms.readthedocs.io/en/latest/index.html
        """

        super().__init__(
            paths=paths,
            prior_passer=prior_passer,
            n_particles=n_particles,
            iters=iters,
            cognitive=cognitive,
            social=social,
            inertia=inertia,
            ftol=ftol,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
        )

        logger.debug("Creating PySwarms NLO")

    def sampler_fom_model_and_fitness(self, model, fitness_function, bounds, init_pos):
        """Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""

        import pyswarms

        return pyswarms.global_best.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=model.prior_count,
            bounds=bounds,
            options={"c1": self.cognitive, "c2": self.social, "w": self.inertia},
            ftol=self.ftol,
            init_pos=init_pos,
        )


class PySwarmsLocal(AbstractPySwarms):
    def __init__(
        self,
        paths=None,
        prior_passer=None,
        n_particles=None,
        iters=None,
        cognitive=None,
        social=None,
        inertia=None,
        number_of_k_neighbors=None,
        minkowski_p_norm=None,
        ftol=None,
        initializer=None,
        iterations_per_update=None,
        number_of_cores=None,
    ):
        """ A PySwarms Particle Swarm Optimizer global non-linear search.

        For a full description of PySwarms, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/pyswarms

        https://pyswarms.readthedocs.io/en/latest/index.html

        A Local-best Particle Swarm Optimization (lbest PSO) algorithm.

        Similar to global-best PSO, it takes a set of candidate solutions, and finds the best solution using a
        position-velocity update method. However, it uses a ring topology, thus making the particles attracted to its
         corresponding neighborhood.

        The position update can be defined as:
        xi(t+1)=xi(t)+vi(t+1)

        Where the position at the current timestep t
        is updated using the computed velocity at t+1

        . Furthermore, the velocity update is defined as:
        vij(t+1)=m∗vij(t)+c1r1j(t)[yij(t)−xij(t)]+c2r2j(t)[y^j(t)−xij(t)]

        However, in local-best PSO, a particle doesn’t compare itself to the overall performance of the swarm.
        Instead, it looks at the performance of its nearest-neighbours, and compares itself with them. In general,
        this kind of topology takes much more time to converge, but has a more powerful explorative feature.

        In this implementation, a neighbor is selected via a k-D tree imported from scipy. Distance are computed with
        either the L1 or L2 distance. The nearest-neighbours are then queried from this k-D tree. They are computed
        for every iteration.

        Extensions:

        - Allows runs to be terminated and resumed from the point it was terminated. This is achieved by outputting the
          necessary results (e.g. the points of the particles) during the model-fit after an input number of iterations.

        - Different options for particle intialization, with the default 'prior' method starting all particles over the
        priors defined by each parameter.

        If you use *PySwarms* as part of a published work, please cite the package following the instructions under the
        *Attribution* section of the GitHub page.

        Parameters
        ----------
        paths : af.Paths
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this non-linear search to a subsequent non-linear search.
        n_particles : int
            The number of particles in the swarm used to sample parameter space.
        iters : int
            The number of iterations that are used to sample parameter space.
        cognitive : float
            The cognitive parameter defining how the PSO particles interact with one another to sample parameter space.
        social : float
            The social parameter defining how the PSO particles interact with one another to sample parameter space.
        inertia : float
            The inertia parameter defining how the PSO particles interact with one another to sample parameter space.
        number_of_k_neighbors : int
            number of neighbors to be considered. Must be a positive integer less than n_particles.
        minkowski_p_norm : int
            The Minkowski p-norm to use. 1 is the sum-of-absolute values (or L1 distance) while 2 is the Euclidean
            (or L2) distance.
        ftol : float
            Relative error in objective_func(best_pos) acceptable for convergence.
        initializer : non_linear.initializer.Initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.

        All remaining attributes are emcee parameters and described at the PySwarms API webpage:

        https://pyswarms.readthedocs.io/en/latest/index.html
        """

        self.number_of_k_neighbors = (
            self._config("search", "number_of_k_neighbors", int)
            if number_of_k_neighbors is None
            else number_of_k_neighbors
        )

        self.minkowski_p_norm = (
            self._config("search", "minkowski_p_norm", int)
            if minkowski_p_norm is None
            else minkowski_p_norm
        )

        super().__init__(
            paths=paths,
            prior_passer=prior_passer,
            n_particles=n_particles,
            iters=iters,
            cognitive=cognitive,
            social=social,
            inertia=inertia,
            ftol=ftol,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
        )

        logger.debug("Creating PySwarms NLO")

    def sampler_fom_model_and_fitness(self, model, fitness_function, bounds, init_pos):
        """Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables."""

        import pyswarms

        return pyswarms.local_best.LocalBestPSO(
            n_particles=self.n_particles,
            dimensions=model.prior_count,
            bounds=bounds,
            options={
                "c1": self.cognitive,
                "c2": self.social,
                "w": self.inertia,
                "k": self.number_of_k_neighbors,
                "p": self.minkowski_p_norm,
            },
            ftol=self.ftol,
            init_pos=init_pos,
        )
