from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.mapper.prior_model.abstract import AbstractPriorModel


class Visualizer:
    """
    Methods associated with visualising analysis, model and data before, during
    or after an optimisation.
    """

    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: AbstractPaths,
        model: AbstractPriorModel,
    ):
        pass

    @staticmethod
    def visualize(
        analysis,
        paths: AbstractPaths,
        instance,
        during_analysis,
    ):
        pass

    @staticmethod
    def visualize_before_fit_combined(
        analysis,
        analyses,
        paths: AbstractPaths,
        model: AbstractPriorModel,
    ):
        pass

    @staticmethod
    def visualize_combined(
        analysis,
        analyses,
        paths: AbstractPaths,
        instance,
        during_analysis,
    ):
        pass
