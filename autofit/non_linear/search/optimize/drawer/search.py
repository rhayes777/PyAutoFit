import numpy as np
from typing import Optional

from autofit.database.sqlalchemy_ import sa

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.search.optimize.abstract_optimize import AbstractOptimizer
from autofit.non_linear.initializer import AbstractInitializer
from autofit.non_linear.search.optimize.drawer.plotter import DrawerPlotter
from autofit.non_linear.samples import Samples, Sample
from autofit.plot.output import Output


class Drawer(AbstractOptimizer):
    __identifier_fields__ = ("total_draws",)

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        initializer: Optional[AbstractInitializer] = None,
        iterations_per_update: int = None,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        A Drawer non-linear search, which simply draws a fixed number of samples from the model uniformly from the
        priors.

        Therefore, it does not seek to determine model parameters which maximize the likelihood or map out the
        posterior of the overall parameter space.

        Whilst this is not the typical use case of a non-linear search, it has certain niche applications, for example:

        - Given a model one can determine how much variation there is in the log likelihood / log posterior values.
          By visualizing this as a histogram one can therefore quantify the behaviour of that
          model's `log_likelihood_function`.

        - If the `log_likelihood_function` of a model is stochastic (e.g. different values of likelihood may be
          computed for an identical model due to randomness in the likelihood evaluation) this search can quantify
          the behaviour of that stochasticity.

        - For advanced modeling tools, for example sensitivity mapping performed via the `Sensitivity` object,
          the `Drawer` search may be sufficient to perform the overall modeling task, without the need of performing
          an actual parameter space search.

        The drawer search itself is performed by simply reusing the functionality of the `AbstractInitializer` object.
        Whereas this is normally used to initialize a non-linear search, for the drawer it performed all log
        likelihood evluations.

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        number_of_cores = 1

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs,
        )

        self.logger.debug("Creating Drawer Search")

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using Drawer and the Analysis class which contains the data and returns the log likelihood from
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

        fitness = Fitness(
            model=model,
            analysis=analysis,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
            convert_to_chi_squared=False
        )

        total_draws = self.config_dict_search["total_draws"]

        self.logger.info(
            f"Performing DrawerSearch for a total of {total_draws} points."
        )

        (
            unit_parameter_lists,
            parameter_lists,
            log_posterior_list,
        ) = self.initializer.samples_from_model(
            total_points=self.config_dict_search["total_draws"],
            model=model,
            fitness=fitness,
        )

        search_internal = {
            "parameter_lists" : parameter_lists,
            "log_posterior_list" : log_posterior_list,
            "time": self.timer.time
        }

        self.paths.save_search_internal(
            obj=search_internal,
        )

        self.logger.info("Drawer complete")

    def samples_from(self, model, search_internal):

        search_internal_dict = self.paths.load_search_internal()

        parameter_lists = search_internal_dict["parameter_lists"]
        log_posterior_list = search_internal_dict["log_posterior_list"]

        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector))
            for vector in parameter_lists
        ]
        log_likelihood_list = [
            lp - prior
            for lp, prior in zip(
                log_posterior_list, log_prior_list
            )
        ]

        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return Samples(
            model=model,
            sample_list=sample_list,
            samples_info=search_internal_dict,
        )

    def plot_results(
        self,
        samples,
        during_analysis : bool = False,
    ):
        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["drawer"][name]

        plotter = DrawerPlotter(
            samples=samples,
            output=Output(
                path=self.paths.image_path / "search", format="png"
            ),
        )
