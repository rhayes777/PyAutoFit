import os
import pickle

import autofit.optimize.non_linear.multi_nest
import autofit.optimize.non_linear.non_linear
from autofit import conf, ModelMapper, convert_paths
from autofit import exc
from autofit.optimize import grid_search
from autofit.tools.promise import PromiseResult


class AbstractPhase:
    @convert_paths
    def __init__(
        self,
        paths,
        *,
        optimizer_class=autofit.optimize.non_linear.multi_nest.MultiNest,
        model=None,
    ):
        """
        A phase in an lensing pipeline. Uses the set non_linear optimizer to try to
        fit_normal models and image passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a non_linear optimizer
        """

        self.paths = paths

        self.optimizer = optimizer_class(self.paths)
        self.model = model or ModelMapper()

    def __str__(self):
        return self.optimizer.paths.phase_name

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.optimizer.paths.phase_name}>"

    @property
    def result(self):
        return PromiseResult(self)

    def run_analysis(self, analysis):
        return self.optimizer.fit(analysis=analysis, model=self.model)

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

    def make_result(self, result, analysis):
        raise NotImplementedError()

    @property
    def phase_name(self):
        return self.paths.phase_name

    def save_optimizer_for_phase(self):
        """
        Save the optimizer associated with the phase as a pickle
        """
        with open(self.paths.make_optimizer_pickle_path(), "w+b") as f:
            f.write(pickle.dumps(self.optimizer))
        with open(self.paths.make_model_pickle_path(), "w+b") as f:
            f.write(pickle.dumps(self.model))

    def save_metadata(self, data_name, pipeline_name):
        """
        Save metadata associated with the phase, such as the name of the pipeline, the
        name of the phase and the name of the dataset being fit
        """
        with open("{}/metadata".format(self.paths.make_path()), "w+") as f:
            f.write(
                "pipeline={}\nphase={}\nsimulator={}".format(
                    pipeline_name, self.paths.phase_name, data_name
                )
            )

    def assert_optimizer_pickle_matches_for_phase(self):
        """
        Assert that the previously saved optimizer is equal to the phase's optimizer if
        a saved optimizer is found.

        Raises
        -------
        exc.PipelineException
        """
        path = self.paths.make_optimizer_pickle_path()
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


class Phase(AbstractPhase):
    @convert_paths
    def __init__(
        self,
        paths,
        *,
        analysis_class,
        optimizer_class=autofit.optimize.non_linear.multi_nest.MultiNest,
        model=None,
    ):
        super().__init__(paths, optimizer_class=optimizer_class, model=model)
        self.analysis_class = analysis_class

    def make_result(self, result, analysis):
        return result

    def make_analysis(self, dataset, results):
        return self.analysis_class(dataset, results)

    def run(self, dataset, results=None):
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

        analysis = self.make_analysis(dataset=dataset, results=results)

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
            number_of_steps=10,
            optimizer_class=autofit.optimize.non_linear.multi_nest.MultiNest,
            **kwargs,
        ):
            super().__init__(paths, optimizer_class=optimizer_class, **kwargs)
            self.optimizer = grid_search.GridSearch(
                paths=self.paths,
                number_of_steps=number_of_steps,
                optimizer_class=optimizer_class,
                parallel=parallel,
            )

        # noinspection PyMethodMayBeStatic,PyUnusedLocal
        def make_result(self, result, analysis):
            return result

        def run_analysis(self, analysis):
            return self.optimizer.fit(analysis, self.model, self.grid_priors)

        @property
        def grid_priors(self):
            raise NotImplementedError(
                "The grid priors property must be implemented to provide a list of "
                "priors to be grid searched"
            )

    return GridSearchExtension
