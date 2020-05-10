import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict

import dill

from autofit import conf, ModelMapper, convert_paths
from autofit import exc
from autofit.mapper.prior.promise import PromiseResult
from autofit.optimize import grid_search
from autofit.optimize.non_linear.emcee import Emcee
from autofit.optimize.non_linear.paths import Paths

logger = logging.getLogger(__name__)


class AbstractPhase:
    @convert_paths
    def __init__(
            self,
            paths: Paths,
            *,
            non_linear_class=Emcee,
            model=None,
    ):
        """
        A phase in an lens pipeline. Uses the set non_linear optimizer to try to
        fit_normal models and image passed to it.

        Parameters
        ----------
        non_linear_class: class
            The class of a non_linear optimizer
        """

        self.paths = paths

        self.optimizer = non_linear_class(paths=self.paths)
        self.model = model or ModelMapper()

        self.pipeline_name = None
        self.pipeline_tag = None
        self.meta_dataset = None

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
            "phase": self.paths.phase_name,
            "phase_tag": self.paths.phase_tag,
            "pipeline": self.pipeline_name,
            "pipeline_tag": self.pipeline_tag,
            "non_linear_search": type(self.optimizer).__name__.lower(),
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
            pickle.dump(
                meta_dataset, f
            )

    def save_phase_attributes(self, phase_attributes):
        with open(
                f"{self.paths.pickle_path}/phase_attributes.pickle",
                "wb+"
        ) as f:
            pickle.dump(
                phase_attributes, f
            )

    def save_info(self, info):
        """
        Save the dataset associated with the phase
        """
        with open("{}/info.pickle".format(self.paths.make_path()), "wb") as f:
            pickle.dump(info, f)

    def __str__(self):
        return self.optimizer.paths.phase_name

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.optimizer.paths.phase_name}>"

    @property
    def result(self) -> PromiseResult:
        """
        A PromiseResult allows promises to be defined, which express the equality
        between posteriors or best fits from this phase and priors or constants
        in some subsequent phase.
        """
        return PromiseResult(self)

    def run_analysis(self, analysis):
        return self.optimizer.full_fit(model=self.model, analysis=analysis)

    def customize_priors(self, results):
        """
        Perform any prior or instance passing. This could involve setting model
        attributes equal to priors or instances from a previous phase.

        Parameters
        ----------
        results: ResultsCollection
            The result of the previous phase
        """
        pass

    def make_phase_attributes(self, analysis):
        raise NotImplementedError()

    def make_result(self, result, analysis):
        raise NotImplementedError()

    @property
    def phase_name(self):
        return self.paths.phase_name

    def save_optimizer_for_phase(self):
        """
        Save the optimizer associated with the phase as a pickle
        """
        with open(self.paths.make_non_linear_pickle_path(), "w+b") as f:
            f.write(pickle.dumps(self.optimizer))
        with open(self.paths.make_model_pickle_path(), "w+b") as f:
            f.write(pickle.dumps(self.model))

    def assert_optimizer_pickle_matches_for_phase(self):
        """
        Assert that the previously saved optimizer is equal to the phase's optimizer if
        a saved optimizer is found.

        Raises
        -------
        exc.PipelineException
        """
        path = self.paths.make_non_linear_pickle_path()
        if os.path.exists(path):
            with open(path, "r+b") as f:
                loaded_optimizer = pickle.loads(f.read())
                if self.optimizer != loaded_optimizer:
                    raise exc.PipelineException(
                        f"Can't restart phase at path {path} because settings don't "
                        f"match. Did you change the optimizer settings?"
                    )

        path = self.paths.make_model_pickle_path()
        if os.path.exists(path):
            with open(path, "r+b") as f:
                loaded_model = pickle.loads(f.read())
                if self.model != loaded_model:
                    raise exc.PipelineException(
                        f"Can't restart phase at path {path} because settings don't "
                        f"match. Did you change the model?"
                    )

    def assert_and_save_pickle(self):
        if conf.instance.general.get("output", "assert_pickle_matches", bool):
            self.assert_optimizer_pickle_matches_for_phase()
        self.save_optimizer_for_phase()


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
            non_linear_class=Emcee,
            model=None,
    ):
        super().__init__(paths, non_linear_class=non_linear_class, model=model)
        self.analysis_class = analysis_class

    def make_result(self, result, analysis):
        return result

    def make_analysis(self, dataset):
        return self.analysis_class(dataset)

    def run(self, dataset: Dataset, results=None, info=None):
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
        self.save_info(info=info)

        analysis = self.make_analysis(dataset=dataset)

        self.customize_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)

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
                number_of_steps=4,
                non_linear_class=Emcee,
                **kwargs,
        ):
            super().__init__(paths, non_linear_class=non_linear_class, **kwargs)
            self.optimizer = grid_search.GridSearch(
                paths=self.paths,
                number_of_steps=number_of_steps,
                non_linear_class=non_linear_class,
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

            return result

        def run_analysis(self, analysis):
            return self.optimizer.fit(model=self.model, analysis=analysis, grid_priors=self.grid_priors)

        @property
        def grid_priors(self):
            raise NotImplementedError(
                "The grid priors property must be implemented to provide a list of "
                "priors to be grid searched"
            )

    return GridSearchExtension
