from abc import ABC
from typing import Optional
import warnings

from autoconf import conf
from autofit.database.sqlalchemy_ import sa
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.initializer import (
    InitializerPrior,
    AbstractInitializer,
    InitializerParamBounds,
)
from autofit.non_linear.samples import SamplesNest
from autofit.non_linear.plot.nest_plotters import NestPlotter
from autofit.non_linear.plot.output import Output


class AbstractNest(NonLinearSearch, ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        iterations_per_update: Optional[int] = None,
        number_of_cores: Optional[int] = None,
        session: Optional[sa.orm.Session] = None,
        initializer: Optional[AbstractInitializer] = None,
        **kwargs
    ):
        """
        Abstract class of a nested sampling `NonLinearSearch` (e.g. MultiNest, Dynesty).

        **PyAutoFit** allows a nested sampler to automatically terminate when the acceptance ratio falls below an input
        threshold value. When this occurs, all samples are accepted using the current maximum log likelihood value,
        irrespective of how well the model actually fits the data.

        This feature should be used for non-linear searches where the nested sampler gets 'stuck', for example because
        the log likelihood function is stochastic or varies rapidly over small scales in parameter space. The results of
        samples using this feature are not realiable (given the log likelihood is being manipulated to end the run), but
        they are still valid results for linking priors to a new search and non-linear search.

        Parameters
        ----------
        session
            An SQLAlchemy session instance so the results of the model-fit are written to an SQLite database.
        """
        if isinstance(initializer, InitializerParamBounds):
            raise ValueError(
                "InitializerParamBounds cannot be used for nested sampling"
            )

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer or InitializerPrior(),
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

    @property
    def config_type(self):
        return conf.instance["non_linear"]["nest"]

    @property
    def samples_cls(self):
        return SamplesNest

    @property
    def plotter_cls(self):
        return NestPlotter

    def plot_results(self, samples):

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["nest"][name]

        plotter = self.plotter_cls(
            samples=samples,
            output=Output(path=self.paths.image_path / "search", format="png"),
        )
        if should_plot("corner_anesthetic"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plotter.corner_anesthetic()