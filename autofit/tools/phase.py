import logging
from os import path
import pickle
from abc import ABC, abstractmethod
from typing import Dict

from autoconf import conf
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior.promise import PromiseResult
from autofit.non_linear.grid import grid_search

logger = logging.getLogger(__name__)


class AbstractPhase:
    def __init__(
            self,
            *,
            search,
            model=None,
    ):
        """
        A phase in an lens pipeline. Uses the set non_linear search to try to
        fit_normal models and image passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        """
        self.search = search
        self.model = model or ModelMapper()

        self.pipeline_name = None
        self.pipeline_tag = None

    @property
    def paths(self):
        return self.search.paths

    @property
    def folders(self):
        return self.search.path_prefix

    @property
    def phase_property_collections(self):
        """
        Returns
        -------
        phase_property_collections: [PhaseProperty]
            A list of phase property collections associated with this phase. This is
            used in automated prior passing and should be overridden for any phase that
            contains its own PhasePropertys.
        """
        return []

    @property
    def _default_metadata(self) -> Dict[str, str]:
        """
        A dictionary of metadata describing this phase, including the pipeline
        that it's embedded in.
        """
        return {
            "phase": self.paths.name,
            "phase_tag": self.paths.tag,
            "pipeline": self.pipeline_name,
            "pipeline_tag": self.pipeline_tag,
        }

    def __str__(self):
        return self.search.paths.name

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.search.paths.name}>"

    def run(self, dataset, mask, results=None):
        raise NotImplementedError()

    def modify_search_paths(self):
        raise NotImplementedError()

    @property
    def result(self) -> PromiseResult:
        """
        A PromiseResult allows promises to be defined, which express the equality
        between posteriors or best fits from this phase and priors or constants
        in some subsequent phase.
        """
        return PromiseResult(self)

    def run_analysis(self, analysis, info=None, pickle_files=None, log_likelihood_cap=None):

        return self.search.fit(model=self.model, analysis=analysis, info=info, pickle_files=pickle_files, log_likelihood_cap=log_likelihood_cap)

    def make_attributes(self, analysis):
        raise NotImplementedError()

    def make_result(self, result, analysis):
        raise NotImplementedError()

    @property
    def name(self):
        return self.paths.name


class Dataset(ABC):
    """
    Comprises the data that is fit by the pipeline. May also contain meta data, noise, PSF, etc.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of this data for use in querying
        """

    @classmethod
    def load(cls, filename) -> "Dataset":
        """
        Load the dataset at the specified filename

        Parameters
        ----------
        filename
            The filename containing the dataset

        Returns
        -------
        The dataset
        """
        with open(filename, "rb") as f:
            return pickle.load(f)


class Phase(AbstractPhase):
    def __init__(
            self,
            *,
            analysis_class,
            search,
            model=None,
    ):
        super().__init__(search=search, model=model)
        self.analysis_class = analysis_class

    def make_result(self, result, analysis):
        return result

    def make_analysis(self, dataset):
        return self.analysis_class(dataset)

    def run(self, dataset: Dataset, results=None, info=None, pickle_files=None, log_likelihood_cap=None):
        """
        Run this phase.

        Parameters
        ----------
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        dataset: scaled_array.ScaledSquarePixelArray
            An masked_imaging that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper_galaxies.
        """

        self.model = self.model.populate(results)

        analysis = self.make_analysis(dataset=dataset)

        result = self.run_analysis(analysis=analysis, info=info, pickle_files=pickle_files, log_likelihood_cap=log_likelihood_cap)

        return self.make_result(result=result, analysis=None)


def as_grid_search(phase_class, parallel=False):
    """
        Returns a grid search phase class from a regular phase class. Instead of the phase
    being optimised by a single non-linear optimiser, a new optimiser is created for
    each square in a grid.

    Parameters
    ----------
    phase_class
        The original phase class
    parallel: bool
        Indicates whether non linear searches in the grid should be performed on
        parallel processes.

    Returns
    -------
    grid_search_phase_class: GridSearchExtension
        A class that inherits from the original class, replacing the optimiser with a
        grid search optimiser.

    """

    class GridSearchExtension(phase_class):
        def __init__(
                self,
                *,
                search,
                number_of_steps=4,
                **kwargs,
        ):
            super().__init__(search=search, **kwargs)

            self.search = grid_search.GridSearch(
                paths=self.paths,
                number_of_steps=number_of_steps,
                search=search,
                parallel=parallel,
            )

        def save_grid_search_result(self, grid_search_result):
            with open(
                    path.join(self.paths.pickle_path, "grid_search_result.pickle"),
                    "wb+"
            ) as f:
                pickle.dump(
                    grid_search_result, f
                )

        # noinspection PyMethodMayBeStatic,PyUnusedLocal
        def make_result(self, result, analysis):
            self.save_grid_search_result(grid_search_result=result)
            open(self.paths.has_completed_path, "w+").close()

            return self.Result(
                samples=result.samples,
                previous_model=result.model,
                analysis=analysis,
                search=self.search,
            )

        def run_analysis(self, analysis, **kwargs):
            self.search.search.paths = self.paths
            self.search.paths = self.paths

            return self.search.fit(model=self.model, analysis=analysis, grid_priors=self.grid_priors)

        @property
        def grid_priors(self):
            raise NotImplementedError(
                "The grid priors property must be implemented to provide a list of "
                "priors to be grid searched"
            )

    return GridSearchExtension


class AbstractSettingsPhase:

    def __init__(self, log_likelihood_cap=None):

        self.log_likelihood_cap = log_likelihood_cap

    @property
    def log_likelihood_cap_tag(self):
        """Generate a bin up tag, to customize phase names based on the resolutioon the image is binned up by for faster \
        run times.

        This changes the phase settings folder is tagged as follows:

        bin_up_factor = 1 -> settings
        bin_up_factor = 2 -> settings_bin_up_factor_2
        bin_up_factor = 2 -> settings_bin_up_factor_2
        """
        if self.log_likelihood_cap is None:
            return ""
        return f"__{conf.instance['notation']['settings_tags']['phase']['log_likelihood_cap']}" \
               + "_{0:.1f}".format(self.log_likelihood_cap)
