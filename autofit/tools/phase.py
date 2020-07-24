import logging
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict
import dill

from autoconf import conf
from autofit.mapper.model_mapper import ModelMapper
from autofit.non_linear.paths import convert_paths
from autofit.mapper.prior.promise import PromiseResult
from autofit.non_linear import grid_search
from autofit.non_linear.paths import Paths

logger = logging.getLogger(__name__)


class AbstractPhase:
    @convert_paths
    def __init__(
            self,
            paths: Paths,
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

        self.search = deepcopy(search)
        self.model = model or ModelMapper()

        self.pipeline_name = None
        self.pipeline_tag = None
        self.meta_dataset = None

    @property
    def paths(self):
        return self.search.paths

    def save_model_info(self):
        """Save the model.info file, which summarizes every parameter and prior."""
        with open(self.paths.file_model_info, "w+") as f:
            f.write(self.model.info)

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

    def make_metadata_text(self, dataset_name):
        return "\n".join(
            f"{key}={value or ''}"
            for key, value
            in {
                **self._default_metadata,
                "dataset_name": dataset_name
            }.items()
        )

    def save_metadata(self, dataset):
        """
        Save metadata associated with the phase, such as the name of the pipeline, the
        name of the phase and the name of the dataset being fit
        """
        with open("{}/metadata".format(self.paths.make_path()), "w+") as f:
            f.write(
                self.make_metadata_text(
                    dataset.name
                )
            )

    def save_dataset(self, dataset):
        """
        Save the dataset associated with the phase
        """
        with open(f"{self.paths.pickle_path}/dataset.pickle", "wb") as f:
            pickle.dump(dataset, f)

    def save_mask(self, mask):
        """
        Save the mask associated with the phase
        """
        with open(f"{self.paths.pickle_path}/mask.pickle", "wb") as f:
            dill.dump(mask, f)

    def save_meta_dataset(self, meta_dataset):
        with open(
                f"{self.paths.pickle_path}/meta_dataset.pickle",
                "wb+"
        ) as f:
            try:
                pickle.dump(
                    meta_dataset, f
                )
            except AttributeError:
                pickle.dump(
                    meta_dataset, f
                )

    def save_settings(self, settings):
        with open(
                f"{self.paths.pickle_path}/settings.pickle",
                "wb+"
        ) as f:
            pickle.dump(
                settings, f
            )

    def save_phase_attributes(self, phase_attributes):
        with open(
                f"{self.paths.pickle_path}/phase_attributes.pickle",
                "wb+"
        ) as f:
            pickle.dump(
                phase_attributes, f
            )

    def __str__(self):
        return self.search.paths.name

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.search.paths.name}>"

    @property
    def result(self) -> PromiseResult:
        """
        A PromiseResult allows promises to be defined, which express the equality
        between posteriors or best fits from this phase and priors or constants
        in some subsequent phase.
        """
        return PromiseResult(self)

    def run_analysis(self, analysis, info=None, pickle_files=None):
        return self.search.fit(model=self.model, analysis=analysis, info=info, pickle_files=pickle_files)

    def make_phase_attributes(self, analysis):
        raise NotImplementedError()

    def make_result(self, result, analysis):
        raise NotImplementedError()

    @property
    def phase_name(self):
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
    @convert_paths
    def __init__(
            self,
            paths,
            *,
            analysis_class,
            search,
            model=None,
    ):
        super().__init__(paths=paths, search=search, model=model)
        self.analysis_class = analysis_class

    def make_result(self, result, analysis):
        return result

    def make_analysis(self, dataset):
        return self.analysis_class(dataset)

    def run(self, dataset: Dataset, results=None, info=None, pickle_files=None):
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
        self.save_metadata(dataset=dataset)
        self.save_dataset(dataset=dataset)

        self.model = self.model.populate(results)

        analysis = self.make_analysis(dataset=dataset)

        result = self.run_analysis(analysis=analysis, info=info, pickle_files=pickle_files)

        return self.make_result(result=result, analysis=None)


def as_grid_search(phase_class, parallel=False):
    """
    Create a grid search phase class from a regular phase class. Instead of the phase
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
        @convert_paths
        def __init__(
                self,
                paths,
                *,
                search,
                number_of_steps=4,
                **kwargs,
        ):

            super().__init__(paths, search=search, **kwargs)

            self.search = grid_search.GridSearch(
                paths=self.paths,
                number_of_steps=number_of_steps,
                search=search,
                parallel=parallel,
            )

        def save_grid_search_result(self, grid_search_result):
            with open(
                    f"{self.paths.pickle_path}/grid_search_result.pickle",
                    "wb+"
            ) as f:
                pickle.dump(
                    grid_search_result, f
                )

        # noinspection PyMethodMayBeStatic,PyUnusedLocal
        def make_result(self, result, analysis):
            self.save_grid_search_result(grid_search_result=result)

            return self.Result(
                samples=result.samples,
                previous_model=result.model,
                analysis=analysis,
                search=self.search,
            )

        def run_analysis(self, analysis, **kwargs):
            return self.search.fit(model=self.model, analysis=analysis, grid_priors=self.grid_priors)

        @property
        def grid_priors(self):
            raise NotImplementedError(
                "The grid priors property must be implemented to provide a list of "
                "priors to be grid searched"
            )

    return GridSearchExtension


class AbstractPhaseSettings:

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
        return (
            "__"
            + conf.instance.tag.get("phase", "log_likelihood_cap", str)
            + "_{0:.1f}".format(self.log_likelihood_cap)
        )