from typing import Optional

from autoconf import cached_property
from autofit.database.sqlalchemy_ import sa

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.mle.abstract_mle import AbstractMLE
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.initializer import AbstractInitializer
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.samples import Samples

import copy
from scipy import optimize
import numpy as np


class AbstractBFGS(AbstractMLE):

    method = None

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        initializer: Optional[AbstractInitializer] = None,
        iterations_per_update: int = None,
        session: Optional[sa.orm.Session] = None,
        **kwargs
    ):
        """
        Abstract wrapper for the BFGS and L-BFGS scipy non-linear searches.

        See the docstrings of the `BFGS` and `LBFGS` classes for a description of the arguments of this class.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            session=session,
            **kwargs
        )

        self.logger.debug(f"Creating {self.method} Search")

    @cached_property
    def config_dict_options(self):
        config_dict = copy.deepcopy(self._class_config["options"])

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
        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
            convert_to_chi_squared=True,
            store_history=self.should_plot_start_point
        )

        try:
            search_internal_dict = self.paths.load_search_internal()

            x0 = search_internal_dict["x0"]
            total_iterations = search_internal_dict["total_iterations"]

            self.logger.info(
                "Resuming LBFGS non-linear search (previous samples found)."
            )

        except (FileNotFoundError, TypeError):

            (
                unit_parameter_lists,
                parameter_lists,
                log_posterior_list,
            ) = self.initializer.samples_from_model(
                total_points=1,
                model=model,
                fitness=fitness,
                paths=self.paths,
                n_cores=self.number_of_cores,
            )

            x0 = np.asarray(parameter_lists[0])

            total_iterations = 0

            self.logger.info(
               f"Starting new {self.method} non-linear search (no previous samples found)."
            )

            self.plot_start_point(
                parameter_vector=x0,
                model=model,
                analysis=analysis,
            )

        maxiter = self.config_dict_options.get("maxiter", 1e8)

        while total_iterations < maxiter:
            iterations_remaining = maxiter - total_iterations

            iterations = min(self.iterations_per_update, iterations_remaining)

            if iterations > 0:
                config_dict_options = self.config_dict_options
                config_dict_options["maxiter"] = iterations

                search_internal = optimize.minimize(
                    fun=fitness.__call__,
                    x0=x0,
                    method=self.method,
                    options=config_dict_options,
                    **self.config_dict_search
                )

                total_iterations += search_internal.nit

                search_internal.log_posterior_list = -0.5 * fitness(
                    parameters=search_internal.x
                )

                if self.should_plot_start_point:

                    search_internal.parameters_history_list = fitness.parameters_history_list
                    search_internal.log_likelihood_history_list = fitness.log_likelihood_history_list

                self.paths.save_search_internal(
                    obj=search_internal,
                )

                x0 = search_internal.x

                if search_internal.nit < iterations:
                    return search_internal

                self.perform_update(
                    model=model,
                    analysis=analysis,
                    during_analysis=True,
                    search_internal=search_internal,
                )

        self.logger.info(f"{self.method} sampling complete.")

        return search_internal

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ):
        """
        Returns a `Samples` object from the LBFGS internal results.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        if search_internal is None:
            search_internal = self.paths.load_search_internal()

        x0 = search_internal.x
        total_iterations = search_internal.nit


        if self.should_plot_start_point:

            parameter_lists = search_internal.parameters_history_list
            log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
            log_likelihood_list = search_internal.log_likelihood_history_list

        else:

            parameter_lists = [list(x0)]
            log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
            log_posterior_list = np.array([search_internal.log_posterior_list])
            log_likelihood_list = [
                lp - prior for lp, prior in zip(log_posterior_list, log_prior_list)
            ]

        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        samples_info = {
            "total_iterations": total_iterations,
            "time": self.timer.time if self.timer else None,
        }

        return Samples(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )


class BFGS(AbstractBFGS):
    """
    The BFGS non-linear search, which wraps the scipy Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm.

    See the docstrings of the `BFGS` and `LBFGS` classes for a description of the arguments of this class.

    For a full description of the scipy BFGS method, checkout its documentation:

    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs

    If you use `BFGS` as part of a published work, please cite the package via scipy following the instructions
    under the *Attribution* section of the GitHub page.

    By default, the BFGS method scipy implementation does not store the history of parameter values and
    log likelihood values during the non-linear search. This is because storing these values can require a large
    amount of memory, in contradiction to the BFGS method's primary advantage of being memory efficient.
    This means that it is difficult to visualize the BFGS method results (e.g. log likelihood vs iteration).

    **PyAutoFit** extends the class with the option of using visualize mode, which stores the history of parameter
    values and log likelihood values during the non-linear search. This allows the results of the BFGS method to be
    visualized after the search has completed, and it is enabled by setting the `visualize` flag to `True`.

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
    number_of_cores: int
        The number of cores sampling is performed using a Python multiprocessing Pool instance.
    session
        An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
    visualize
        If True, visualization of the search is enabled, which requires storing the history of parameter values and
        log likelihood values during the non-linear search.
    """

    method = "BFGS"


class LBFGS(AbstractBFGS):
    """
    The L-BFGS non-linear search, which wraps the scipy Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
    algorithm.

    See the docstrings of the `BFGS` and `LBFGS` classes for a description of the arguments of this class.

    For a full description of the scipy L-BFGS method, checkout its documentation:

    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

    If you use `LBFGS` as part of a published work, please cite the package via scipy following the instructions
    under the *Attribution* section of the GitHub page.

    By default, the L-BFGS method scipy implementation does not store the history of parameter values and
    log likelihood values during the non-linear search. This is because storing these values can require a large
    amount of memory, in contradiction to the L-BFGS method's primary advantage of being memory efficient.
    This means that it is difficult to visualize the L-BFGS method results (e.g. log likelihood vs iteration).

    **PyAutoFit** extends the class with the option of using visualize mode, which stores the history of parameter
    values and log likelihood values during the non-linear search. This allows the results of the L-BFGS method to be
    visualized after the search has completed, and it is enabled by setting the `visualize` flag to `True`.

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
    number_of_cores: int
        The number of cores sampling is performed using a Python multiprocessing Pool instance.
    session
        An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
    visualize
        If True, visualization of the search is enabled, which requires storing the history of parameter values and
        log likelihood values during the non-linear search.
    """

    method = "L-BFGS-B"