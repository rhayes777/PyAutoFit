import copy
from os import path
from typing import Optional

from autofit.database.sqlalchemy_ import sa

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.nest import abstract_nest
from autofit.non_linear.nest.abstract_nest import AbstractNest
from autofit.non_linear.nest.ultranest.samples import UltraNestSamples
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
            name: str = "",
            path_prefix: str = "",
            unique_tag: Optional[str] = None,
            prior_passer: PriorPasser = None,
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
        prior_passer
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        iterations_per_update
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores
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
            iterations_per_update=iterations_per_update,
            session=session,
            **kwargs
        )

        self.number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        for key, value in self.config_dict_stepsampler.items():
            setattr(self, key, value)
            if self.config_dict_stepsampler["stepsampler_cls"] is None:
                self.nsteps = None

        self.logger.debug("Creating UltraNest Search")

    @property
    def config_dict_stepsampler(self):

        config_dict = copy.copy(self.config_type[self.__class__.__name__]["stepsampler"]._dict)

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

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

    class Fitness(AbstractNest.Fitness):
        @property
        def resample_figure_of_merit(self):
            """If a sample raises a FitException, this value is returned to signify that the point requires resampling or
             should be given a likelihood so low that it is discard.

             -np.inf is an invalid sample value for Dynesty, so we instead use a large negative number."""
            return -1.0e99

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None):
        """
        Fit a model using Dynesty and the Analysis class which contains the data and returns the log likelihood from
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

        import ultranest

        fitness_function = self.fitness_function_from_model_and_analysis(
            model=model, analysis=analysis, log_likelihood_cap=log_likelihood_cap,
        )

        def prior_transform(cube):
            return model.vector_from_unit_vector(
                unit_vector=cube,
                ignore_prior_limits=True
            )

        sampler = ultranest.ReactiveNestedSampler(
            param_names=model.parameter_names,
            loglike=fitness_function.__call__,
            transform=prior_transform,
            log_dir=self.paths.samples_path,
            **self.config_dict_search
        )

        sampler.stepsampler = self.stepsampler

        finished = False

        while not finished:

            try:
                total_iterations = sampler.ncall
            except AttributeError:
                total_iterations = 0

            if self.config_dict_run["max_ncalls"] is not None:
                iterations = self.config_dict_run["max_ncalls"]
            else:
                iterations = total_iterations + self.iterations_per_update

            if iterations > 0:

                config_dict_run = self.config_dict_run
                config_dict_run.pop("max_ncalls")
                config_dict_run["dKL"] = config_dict_run.pop("dkl")
                config_dict_run["Lepsilon"] = config_dict_run.pop("lepsilon")
                config_dict_run["update_interval_ncall"] = iterations

                sampler.run(
                    max_ncalls=iterations,
                    **config_dict_run
                )

            self.paths.save_object(
                "results",
                sampler.results
            )

            self.perform_update(model=model, analysis=analysis, during_analysis=True)

            iterations_after_run = sampler.ncall

            if (
                    total_iterations == iterations_after_run
                    or iterations_after_run == self.config_dict_run["max_ncalls"]
            ):
                finished = True

    def samples_from(self, model: AbstractPriorModel):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For MulitNest, this requires us to load:

            - The parameter samples, log likelihood values and weight_list from the ultranest.txt file.
            - The total number of samples (e.g. accepted + rejected) from resume.dat.
            - The log evidence of the model-fit from the ultranestsummary.txt file (if this is not yet estimated a
              value of -1.0e99 is used.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        try:

            results_internal = self.paths.load_object(
                "results"
            )

        except FileNotFoundError:

            samples = self.paths.load_object(
                "samples"
            )
            results_internal = samples.results

        return UltraNestSamples.from_results_internal(
            results_internal=results_internal,
            model=model,
            number_live_points=self.config_dict_run["min_num_live_points"],
            unconverged_sample_size=1,
            time=self.timer.time,
        )

    def plot_results(self, samples):

        if not samples.pdf_converged:
            return

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["ultranest"][name]

        plotter = UltraNestPlotter(
            samples=samples,
            output=Output(path=path.join(self.paths.image_path, "search"), format="png")
        )

        if should_plot("cornerplot"):
            plotter.cornerplot()

        if should_plot("runplot"):
            plotter.runplot()

        if should_plot("traceplot"):
            plotter.traceplot()
