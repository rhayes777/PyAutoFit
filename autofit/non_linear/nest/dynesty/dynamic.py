from typing import Optional

from dynesty.dynesty import DynamicNestedSampler
from autofit.non_linear.nest.dynesty.samples import SamplesDynesty

from .abstract import AbstractDynesty, prior_transform


class DynestyDynamic(AbstractDynesty):
    __identifier_fields__ = (
        "bound",
        "sample",
        "enlarge",
        "bootstrap",
        "walks",
        "facc",
        "slices",
        "fmove",
        "max_move"
    )

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            prior_passer=None,
            iterations_per_update: int = None,
            number_of_cores: int = None,
            **kwargs
    ):
        """
        A Dynesty non-linear search, using a dynamically changing number of live points.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty
        https://dynesty.readthedocs.io/en/latest/index.html

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
            number_of_cores=number_of_cores,
            **kwargs
        )

        self.logger.debug("Creating DynestyDynamic Search")

    def samples_from(self, model):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Dynesty, all information that we need is available from the instance of the dynesty sampler.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """
        sampler = DynamicNestedSampler.restore(self.checkpoint_file)

        return SamplesDynesty.from_results_internal(
            model=model,
            results_internal=sampler.results,
            number_live_points=self.total_live_points,
            unconverged_sample_size=1,
            time=self.timer.time,
        )

    def sampler_from(
            self,
            model,
            fitness_function,
            pool,
            queue_size
    ):
        """
        Returns an instance of the Dynesty dynamic sampler set up using the input variables of this class.

        If no existing dynesty sampler exist on hard-disk (located via a `checkpoint_file`) a new instance is
        created with which sampler is performed. If one does exist, the dynesty `restore()` function is used to
        create the instance of the sampler.

        Dynesty samplers with a multiprocessing pool may be created by inputting a dynesty `Pool` object, however
        non pooled instances can also be created by passing `pool=None` and `queue_size=None`.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        fitness_function
            An instance of the fitness class used to evaluate the likelihood of each model.
        pool
            A dynesty Pool object which performs likelihood evaluations over multiple CPUs.
        queue_size
            The number of CPU's over which multiprocessing is performed, determining how many samples are stored
            in the dynesty queue for samples.
        """

        try:

            sampler = DynamicNestedSampler.restore(
                fname=self.checkpoint_file,
                pool=pool
            )

            self.check_pool(sampler=sampler, pool=pool)

            return sampler

        except FileNotFoundError:

            if pool is not None:

                return DynamicNestedSampler(
                    loglikelihood=pool.loglike,
                    prior_transform=pool.prior_transform,
                    ndim=model.prior_count,
                    queue_size=queue_size,
                    pool=pool,
                    **self.config_dict_search
                )

            return DynamicNestedSampler(
                loglikelihood=fitness_function,
                prior_transform=prior_transform,
                ndim=model.prior_count,
                logl_args=[model, fitness_function],
                ptform_args=[model],
                **self.config_dict_search
            )

    @property
    def total_live_points(self):
        return self.config_dict_run["nlive_init"]
