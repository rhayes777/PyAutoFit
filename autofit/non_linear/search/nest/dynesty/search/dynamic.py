from __future__ import annotations
from typing import Optional

from dynesty.dynesty import DynamicNestedSampler
from autofit.mapper.prior_model.abstract import AbstractPriorModel

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
        iterations_per_update
            The number of iterations performed between update (e.g. output latest model to hard-disk, visualization).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            **kwargs
        )

        self.logger.debug("Creating DynestyDynamic Search")

    @property
    def search_internal(self):
        return DynamicNestedSampler.restore(self.checkpoint_file)

    def search_internal_from(
            self,
            model: AbstractPriorModel,
            fitness,
            checkpoint_exists : bool,
            pool: Optional,
            queue_size: Optional[int]
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
        fitness
            An instance of the fitness class used to evaluate the likelihood of each model.
        pool
            A dynesty Pool object which performs likelihood evaluations over multiple CPUs.
        queue_size
            The number of CPU's over which multiprocessing is performed, determining how many samples are stored
            in the dynesty queue for samples.
        """

        try:

            search_internal = DynamicNestedSampler.restore(
                fname=self.checkpoint_file,
                pool=pool
            )

            uses_pool = self.read_uses_pool()

            self.check_pool(uses_pool=uses_pool, pool=pool)

            return search_internal

        except (FileNotFoundError, TypeError):

            if pool is not None:

                self.write_uses_pool(uses_pool=True)

                return DynamicNestedSampler(
                    loglikelihood=pool.loglike,
                    prior_transform=pool.prior_transform,
                    ndim=model.prior_count,
                    queue_size=queue_size,
                    pool=pool,
                    **self.config_dict_search
                )

            self.write_uses_pool(uses_pool=False)

            return DynamicNestedSampler(
                loglikelihood=fitness,
                prior_transform=prior_transform,
                ndim=model.prior_count,
                logl_args=[model, fitness],
                ptform_args=[model],
                **self.config_dict_search
            )

    @property
    def number_live_points(self):
        return self.config_dict_run["nlive_init"]