from typing import Optional

from autofit.database.sqlalchemy_ import sa
from autofit.non_linear.optimize.pyswarms.abstract import AbstractPySwarms


class PySwarmsLocal(AbstractPySwarms):
    __identifier_fields__ = (
        "n_particles",
        "cognitive",
        "social",
        "inertia",
        "number_of_k_neighbors",
        "minkowski_p_norm"
    )

    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            prior_passer=None,
            iterations_per_update: int = None,
            number_of_cores: int = None,
            session: Optional[sa.orm.Session] = None,
            **kwargs
    ):
        """
        A PySwarms Particle Swarm Optimizer global non-linear search.

        For a full description of PySwarms, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/pyswarms

        https://pyswarms.readthedocs.io/en/latest/index.html

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
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

        self.logger.debug("Creating PySwarms Search")

    def sampler_from(self, model, fitness_function, bounds, init_pos):
        """
        Get the static Dynesty sampler which performs the non-linear search, passing it all associated input Dynesty
        variables.
        """

        import pyswarms

        options = {
            "c1": self.config_dict_search["cognitive"],
            "c2": self.config_dict_search["social"],
            "w": self.config_dict_search["inertia"],
            "k": self.config_dict_search["number_of_k_neighbors"],
            "p": self.config_dict_search["minkowski_p_norm"],
        }

        config_dict = self.config_dict_search
        config_dict.pop("cognitive")
        config_dict.pop("social")
        config_dict.pop("inertia")
        config_dict.pop("number_of_k_neighbors")
        config_dict.pop("minkowski_p_norm")

        return pyswarms.local_best.LocalBestPSO(
            dimensions=model.prior_count,
            bounds=bounds,
            init_pos=init_pos,
            options=options,
            **config_dict
        )
