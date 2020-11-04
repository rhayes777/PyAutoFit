import autofit as af
from howtofit.chapter_2_results.src.dataset.dataset import (
    Dataset,
    MaskedDataset,
)
from howtofit.chapter_2_results.src.phase.result import Result
from howtofit.chapter_2_results.src.phase.analysis import Analysis

# The `phase.py` module is mostly unchanged from the previous tutorial, however the `run` function has been updated.


class Phase(af.AbstractPhase):

    profiles = af.PhaseProperty("profiles")

    Result = Result

    def __init__(self, *, profiles, settings, search):
        """
        A phase which fits a model composed of multiple profiles (Gaussian, Exponential) using a `NonLinearSearch`.

        Parameters
        ----------
        profiles : [profiles.Profile]
            The model components (e.g. Gaussian, Exponential) fitted by this phase.
        search: class
            The class of a non_linear search
        data_trim_left : int or None
            The number of pixels by which the data is trimmed from the left-hand side.
        data_trim_right : int or None
            The number of pixels by which the data is trimmed from the right-hand side.
        """
        super().__init__(search=search)

        self.profiles = profiles
        self.settings = settings

    def run(self, dataset: Dataset, mask, info=None):
        """
        Pass a `Dataset` to the phase, running the phase and `NonLinearSearch`.

        Parameters
        ----------
        dataset: aa.Dataset
            The `Dataset` fitted by the phase, as defined in the `dataset.py` module.
        mask: Mask2D
            The mask used for the analysis.

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising information on the `NonLinearSearch` and the maximum likelihood model.
        """

        self.modify_search_paths()

        analysis = self.make_analysis(dataset=dataset, mask=mask)

        result = self.run_analysis(analysis=analysis, info=info)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, mask):
        """
        Returns an Analysis object, which creates the `Dataset` and contains the functions which perform the fit.

        Parameters
        ----------
        dataset: aa.Dataset
            The `Dataset` fitted by the phase, as defined in the `dataset.py` module.

        Returns
        -------
        analysis : Analysis
            An analysis object that the `NonLinearSearch` calls to determine the fit log_likelihood for a given model
            instance.
        """

        masked_dataset = MaskedDataset(
            dataset=dataset, mask=mask, settings=self.settings.settings_masked_dataset
        )

        return Analysis(
            masked_dataset=masked_dataset,
            settings=self.settings,
            image_path=self.search.paths.image_path,
        )

    def make_result(self, result, analysis):
        return self.Result(
            samples=result.samples,
            previous_model=self.model,
            search=self.search,
            analysis=analysis,
        )

    def modify_search_paths(self):
        """
        Modify the output paths of the phase before the non-linear search is run, so that the output path can be
        customized using the tags of the phase.
        """

        self.search.paths.tag = self.settings.tag
