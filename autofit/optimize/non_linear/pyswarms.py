import logging
import os
import pyswarms
import numpy as np
import pickle

from autofit import exc
from autofit.text import samples_text
from autofit.optimize.non_linear import samples
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.paths import Paths

logger = logging.getLogger(__name__)


class PySwarmsGlobal(NonLinearOptimizer):
    def __init__(
        self,
        paths=None,
        sigma=3,
        n_particles=None,
        iters=None,
        cognitive=None,
        social=None,
        inertia=None,
        ftol=None,
        initialize_method=None,
        initialize_ball_lower_limit=None,
        initialize_ball_upper_limit=None,
        iterations_per_update=None,
        number_of_cores=None,
    ):
        """ Class to setup and run a PySwarms Particle Swarm Optimizer global non-linear search.

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
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search samples,
            backups, etc.
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
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
        initialize_method : str
            The method used to generate where walkers are initialized in parameter space, with options:
            ball (default):
                Walkers are initialized by randomly drawing unit values from a uniform distribution between the
                initialize_ball_lower_limit and initialize_ball_upper_limit values. It is recommended these limits are
                small, such that all walkers begin close to one another.
            prior:
                Walkers are initialized by randomly drawing unit values from a uniform distribution between 0 and 1,
                thus being distributed over the prior.
        initialize_ball_lower_limit : float
            The lower limit of the uniform distribution unit values are drawn from when initializing walkers using the
            ball method.
        initialize_ball_upper_limit : float
            The upper limit of the uniform distribution unit values are drawn from when initializing walkers using the
            ball method.
        number_of_cores : int
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.

        All remaining attributes are emcee parameters and described at the PySwarms API webpage:

        https://pyswarms.readthedocs.io/en/latest/index.html
        """

        if paths is None:
            paths = Paths(non_linear_name=type(self).__name__.lower())

        super().__init__(
            paths=paths,
            initialize_method=initialize_method,
            initialize_ball_lower_limit=initialize_ball_lower_limit,
            initialize_ball_upper_limit=initialize_ball_upper_limit
        )

        self.sigma = sigma

        self.n_particles = self.config("search", "n_particles", int) if n_particles is None else n_particles
        self.iters = self.config("search", "iters", int) if iters is None else iters

        self.cognitive = self.config("search", "cognitive", float) if cognitive is None else cognitive
        self.social = self.config("search", "social", float) if social is None else social
        self.inertia = self.config("search", "inertia", float) if inertia is None else inertia
        self.ftol = self.config("search", "ftol", float) if ftol is None else ftol

        self.iterations_per_update = (
            self.config("settings", "iterations_per_update", int)
            if iterations_per_update is None
            else iterations_per_update
        )

        self.number_of_cores = (
            self.config("parallel", "number_of_cores", int)
            if number_of_cores is None
            else number_of_cores
        )

        logger.debug("Creating Emcee NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        """Copy this instance of the emcee non-linear search with all associated attributes.

        This is used to set up the non-linear search on phase extensions."""
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma = self.sigma
        copy.n_particles = self.n_particles
        copy.iters = self.iters
        copy.cognitive = self.cognitive
        copy.social = self.social
        copy.intertia = self.inertia
        copy.ftol = self.ftol
        copy.initialize_method = self.initialize_method
        copy.initialize_ball_lower_limit = self.initialize_ball_lower_limit
        copy.initialize_ball_upper_limit = self.initialize_ball_upper_limit
        copy.iterations_per_upddate = self.iterations_per_update
        copy.number_of_cores = self.number_of_cores

        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __call__(self, params):

            log_likelihoods = []

            for params_of_particle in params:

                try:

                    instance = self.model.instance_from_vector(vector=list(params_of_particle))
                    log_likelihoods.append(-2.0*self.fit_instance(instance))

                except exc.FitException:

                    log_likelihoods.append(-np.inf)

            return np.asarray(log_likelihoods)

    def _fit(self, model, analysis):
        """
        Fit a model using emcee and a function that returns a log likelihood from instances of that model.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        fitness_function
            A function that fits this model to the data, returning the log likelihood of the fit.

        Returns
        -------
        A result object comprising the best-fit model instance, log_likelihood and an *Output* class that enables analysis
        of the full chains used by the fit.
        """
        pool, pool_ids = self.make_pool()

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis, pool_ids=pool_ids,
        )

        if os.path.exists("{}/{}.pickle".format(self.paths.samples_path, "points")):

            init_pos = self.load_initial_points
            total_iterations = self.load_total_iterations

        else:

            init_pos = self.initial_points_from_model(number_of_points=self.n_particles, model=model)
            total_iterations = 0

        finished = False

        logger.info("Running PySwarmsGlobal Optimizer...")

        while not finished:

            pso = pyswarms.global_best.GlobalBestPSO(
                n_particles=self.n_particles,
                dimensions=model.prior_count,
                options={'c1' : self.cognitive, 'c2' : self.social, 'w' : self.inertia},
                ftol=self.ftol,
                init_pos=init_pos,
            )

            iterations_remaining = self.iters - total_iterations

            if self.iterations_per_update > iterations_remaining:
                iterations = iterations_remaining
            else:
                iterations = self.iterations_per_update

            if iterations > 0:

                result = pso.optimize(
                    objective_func=fitness_function.__call__,
                    iters=iterations,
                    n_processes=self.number_of_cores
                )

                total_iterations += iterations

                with open(f"{self.paths.samples_path}/total_iterations.pickle", "wb") as f:
                    pickle.dump(total_iterations, f)

                with open(f"{self.paths.samples_path}/points.pickle", "wb") as f:
                    pickle.dump(pso.pos_history[-1], f)

                with open(f"{self.paths.samples_path}/max_log_likelihood.pickle", "wb") as f:
                    pickle.dump(-0.5*result[0], f)

                with open(f"{self.paths.samples_path}/max_log_likelihood_vector.pickle", "wb") as f:
                    pickle.dump(result[1], f)

            if total_iterations >= self.iters:
                finished = True

        logger.info("PySwarmsGlobal complete")

        self.paths.backup()

        samples = self.samples_from_model(model=model)

        analysis.visualize(
            instance=samples.max_log_likelihood_instance, during_analysis=False
        )

        samples_text.results_to_file(
            samples=samples, file_results=self.paths.file_results, during_analysis=False
        )

        self.paths.backup_zip_remove()

        return Result(samples=samples, previous_model=model)

    @property
    def load_total_iterations(self):
        with open("{}/{}.pickle".format(self.paths.samples_path, "total_iterations"), "rb") as f:
            return pickle.load(f)

    @property
    def load_initial_points(self):
        with open("{}/{}.pickle".format(self.paths.samples_path, "points"), "rb") as f:
            return pickle.load(f)

    @property
    def load_max_log_likelihood(self):
        with open("{}/{}.pickle".format(self.paths.samples_path, "max_log_likelihood"), "rb") as f:
            return pickle.load(f)

    @property
    def load_max_log_likelihood_vector(self):
        with open("{}/{}.pickle".format(self.paths.samples_path, "max_log_likelihood_vector"), "rb") as f:
            return pickle.load(f)

    def fitness_function_from_model_and_analysis(self, model, analysis, pool_ids=None):

        return PySwarmsGlobal.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from_model,
            pool_ids=pool_ids,
        )

    def samples_from_model(self, model):
        """Create a *Samples* object from this non-linear search's output files on the hard-disk and model.

        For Emcee, all quantities are extracted via the hdf5 backend of results.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search chains,
            backups, etc.
        """
        return samples.OptimizerSamples(
            model=model,
            parameters=[list(self.load_max_log_likelihood_vector)],
            log_likelihoods=[self.load_max_log_likelihood]
        )
