import os
from typing import Optional

from autofit.database.sqlalchemy_ import sa

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.nest import abstract_nest
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.nest import SamplesNest
from autofit.plot import UltraNestPlotter
from autofit.plot.output import Output

class UltraNest(abstract_nest.AbstractNest):
    __identifier_fields__ = (
        "draw_multiple",
        "ndraw_min",
        "ndraw_max",
        "min_num_live_points",
        "cluster_num_live_points",
        "insertion_test_zscore_threshold",
        "stepsampler_cls",
        "nsteps"
    )

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            iterations_per_update: int = None,
            number_of_cores: int = None,
            session: Optional[sa.orm.Session] = None,
            **kwargs
    ):
        """
        An UltraNest non-linear search.

        UltraNest is an optional requirement and must be installed manually via the command `pip install ultranest`.
        It is optional as it has certain dependencies which are generally straight forward to install (e.g. Cython).

        For a full description of UltraNest and its Python wrapper PyUltraNest, checkout its Github and documentation
        webpages:

        https://github.com/JohannesBuchner/UltraNest
        https://johannesbuchner.github.io/UltraNest/readme.html

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        iterations_per_update
            The number of iterations performed between update (e.g. output latest model to hard-disk, visualization).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

        for key, value in self.config_dict_stepsampler.items():
            setattr(self, key, value)
            if self.config_dict_stepsampler["stepsampler_cls"] is None:
                self.nsteps = None

        self.logger.debug("Creating UltraNest Search")

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using the search and the Analysis class which contains the data and returns the log likelihood from
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
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted ssamples of the fit.
        """

        try:
            import ultranest
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "\n--------------------\n"
                "You are attempting to perform a model-fit using UltraNest. \n\n"
                "However, the optional library UltraNest (https://johannesbuchner.github.io/UltraNest/index.html) is "
                "not installed.\n\n"
                "Install it via the command `pip install ultranest==3.6.2`.\n\n"
                "----------------------"
            )

        fitness = Fitness(
            model=model,
            analysis=analysis,
            fom_is_log_likelihood=True,
            resample_figure_of_merit=-1.0e99
        )

        def prior_transform(cube):
            return model.vector_from_unit_vector(
                unit_vector=cube,
                ignore_prior_limits=True
            )

        log_dir = self.paths.search_internal_path

        try:
            checkpoint_exists = os.path.exists(log_dir / "chains")
        except TypeError:
            checkpoint_exists = False

        if checkpoint_exists:
            self.logger.info(
                "Resuming UltraNest non-linear search (previous samples found)."
            )
        else:
            self.logger.info(
                "Starting new UltraNest non-linear search (no previous samples found)."
            )

        search_internal = ultranest.ReactiveNestedSampler(
            param_names=model.parameter_names,
            loglike=fitness.__call__,
            transform=prior_transform,
            log_dir=log_dir,
            **self.config_dict_search
        )

        search_internal.stepsampler = self.stepsampler

        finished = False

        while not finished:

            try:
                total_iterations = search_internal.ncall
            except AttributeError:
                total_iterations = 0

            if self.config_dict_run["max_ncalls"] is not None:
                iterations = self.config_dict_run["max_ncalls"]
            else:
                iterations = total_iterations + self.iterations_per_update

            if iterations > 0:

                filter_list = ["max_ncalls", "dkl", "lepsilon"]
                config_dict_run = {
                    key: value for key, value
                    in self.config_dict_run.items()
                    if key
                    not in filter_list
                }

                config_dict_run["update_interval_ncall"] = iterations

                search_internal.run(
                    max_ncalls=iterations,
                    **config_dict_run
                )

            self.paths.save_search_internal(
                  obj=search_internal.results,
              )

            iterations_after_run = search_internal.ncall

            if (
                    total_iterations == iterations_after_run
                    or iterations_after_run == self.config_dict_run["max_ncalls"]
            ):
                finished = True

            if not finished:

                self.perform_update(
                    model=model,
                    analysis=analysis,
                    during_analysis=True,
                    search_internal=search_internal
                )

        return search_internal

    def samples_info_from(self, search_internal=None):

        search_internal = search_internal or self.paths.load_search_internal()

        return {
            "log_evidence": search_internal["logz"],
            "total_samples": search_internal["ncall"],
            "time": self.timer.time if self.timer else None,
            "number_live_points": self.config_dict_run["min_num_live_points"]
        }

    def samples_via_internal_from(self, model: AbstractPriorModel, search_internal=None):
        """
        Returns a `Samples` object from the ultranest internal results.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        search_internal = search_internal.results or self.paths.load_search_internal()

        parameters = search_internal["weighted_samples"]["points"]
        log_likelihood_list = search_internal["weighted_samples"]["logl"]
        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameters
        ]
        weight_list = search_internal["weighted_samples"]["weights"]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return SamplesNest(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info_from(search_internal=search_internal),
            search_internal=search_internal,
        )

    def config_dict_with_test_mode_settings_from(self, config_dict):

        return {
            **config_dict,
            "max_iters": 1,
            "max_ncalls": 1,
        }

    @property
    def config_dict_stepsampler(self):

        config_dict = {}

        config_dict_step = self.config_type[self.__class__.__name__]["stepsampler"]

        for key, value in config_dict_step.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                config_dict[key] = value

        return config_dict

    @property
    def stepsampler(self):

        from ultranest import stepsampler

        config_dict_stepsampler = self.config_dict_stepsampler
        stepsampler_cls = config_dict_stepsampler["stepsampler_cls"]
        config_dict_stepsampler.pop("stepsampler_cls")

        if stepsampler_cls is None:
            return None
        elif stepsampler_cls == "RegionMHSampler":
            return stepsampler.RegionMHSampler(**config_dict_stepsampler)
        elif stepsampler_cls == "AHARMSampler":
            config_dict_stepsampler.pop("scale")
            return stepsampler.AHARMSampler(**config_dict_stepsampler)
        elif stepsampler_cls == "CubeMHSampler":
            return stepsampler.CubeMHSampler(**config_dict_stepsampler)
        elif stepsampler_cls == "CubeSliceSampler":
            return stepsampler.CubeSliceSampler(**config_dict_stepsampler)
        elif stepsampler_cls == "RegionSliceSampler":
            return stepsampler.RegionSliceSampler(**config_dict_stepsampler)

    def plot_results(self, samples):

        if not samples.pdf_converged:
            return

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["ultranest"][name]

        plotter = UltraNestPlotter(
            samples=samples,
            output=Output(self.paths.image_path / "search", format="png")
        )

        if should_plot("cornerplot"):
            plotter.cornerplot()

        if should_plot("runplot"):
            plotter.runplot()

        if should_plot("traceplot"):
            plotter.traceplot()