from __future__ import annotations

from autoconf import cached_property

from pathlib import Path
from typing import Optional, Union

from dynesty import NestedSampler as StaticSampler

from autofit.database.sqlalchemy_ import sa
from autofit.mapper.prior_model.abstract import AbstractPriorModel

from .abstract import AbstractDynesty, prior_transform


class GradWrapper:
    def __init__(self, function):
        self.function = function

    @cached_property
    def grad(self):
        import jax
        from jax import grad

        print("Compiling gradient")
        return jax.jit(grad(self.function))

    def __getstate__(self):
        return {"function": self.function}

    def __setstate__(self, state):
        self.__init__(state["function"])

    def __call__(self, *args, **kwargs):
        return self.grad(*args, **kwargs)


class DynestyStatic(AbstractDynesty):
    __identifier_fields__ = (
        "nlive",
        "bound",
        "sample",
        "bootstrap",
        "enlarge",
        "walks",
        "facc",
        "slices",
        "fmove",
        "max_move",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[Union[str, Path]] = None,
        unique_tag: Optional[str] = None,
        iterations_per_update: int = None,
        number_of_cores: int = None,
        session: Optional[sa.orm.Session] = None,
        use_gradient: bool = False,
        **kwargs,
    ):
        """
        A Dynesty `NonLinearSearch` using a static number of live points.

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
        use_gradient
            Determines whether the gradient should be passed to the Dynesty sampler.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs,
        )

        self.use_gradient = use_gradient

        self.logger.debug("Creating DynestyStatic Search")

    @property
    def search_internal(self):
        return StaticSampler.restore(self.checkpoint_file)

    def search_internal_from(
        self,
        model: AbstractPriorModel,
        fitness,
        checkpoint_exists: bool,
        pool: Optional,
        queue_size: Optional[int],
    ):
        """
        Returns an instance of the Dynesty static sampler set up using the input variables of this class.

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
        if self.use_gradient:
            gradient = GradWrapper(fitness)
        else:
            gradient = None

        if checkpoint_exists:
            search_internal = StaticSampler.restore(
                fname=self.checkpoint_file, pool=pool
            )

            uses_pool = self.read_uses_pool()

            self.check_pool(uses_pool=uses_pool, pool=pool)

            return search_internal

        else:
            live_points = self.live_points_init_from(model=model, fitness=fitness)

            if pool is not None:
                self.write_uses_pool(uses_pool=True)
                return StaticSampler(
                    loglikelihood=pool.loglike,
                    gradient=gradient,
                    prior_transform=pool.prior_transform,
                    ndim=model.prior_count,
                    live_points=live_points,
                    queue_size=queue_size,
                    pool=pool,
                    **self.config_dict_search,
                )

            self.write_uses_pool(uses_pool=False)
            return StaticSampler(
                loglikelihood=fitness,
                gradient=gradient,
                prior_transform=prior_transform,
                ndim=model.prior_count,
                logl_args=[model, fitness],
                ptform_args=[model],
                live_points=live_points,
                **self.config_dict_search,
            )

    @property
    def number_live_points(self):
        return self.config_dict_search["nlive"]
