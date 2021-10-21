from typing import Optional

from dynesty.dynesty import DynamicNestedSampler

from .abstract import AbstractDynesty, prior_transform


class DynestyDynamic(AbstractDynesty):
    __identifier_fields__ = (
        "bound",
        "sample",
        "enlarge",
        "bootstrap",
        "vol_dec",
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

    def sampler_from(
            self,
            model,
            fitness_function,
            pool=None
    ):
        """
        Get the dynamic Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables.
        """

        live_points = self.live_points_from_model_and_fitness_function(
            model=model, fitness_function=fitness_function
        )

        return DynamicNestedSampler(
            loglikelihood=fitness_function,
            prior_transform=prior_transform,
            ndim=model.prior_count,
            logl_args=[model, fitness_function],
            ptform_args=[model],
            live_points=live_points,
            queue_size=self.number_of_cores,
            pool=pool,
            **self.config_dict_search
        )

    @property
    def total_live_points(self):
        return self.config_dict_run["nlive_init"]
