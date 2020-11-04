import autofit as af
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.dataset.dataset import (
    Dataset,
    MaskedDataset,
)
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.phase.result import (
    Result,
)
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.phase.analysis import (
    Analysis,
)
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.phase.settings import (
    SettingsPhase,
)


"""
The phase module has new features not included in tutorial 6, which customize the `Dataset` that is fitted and tag
the output path of the results.
"""


class Phase(af.AbstractPhase):

    """
    Because we now have multiple profiles in our model, we have renamed `gaussian` to `profiles`. As before,
    PyAutoFit uses this information to map the input Profile classes to a model instance when performing a fit.
    """

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
        settings : SettingsPhase
            The collection of settings of the phase used to augment the data that is fitted and tag the output path.
        """

        """
        Here, we create a `tag` for our phase. If we use an optional phase setting to alter the `Dataset` we fit (here,
        a data_trim_ variable), we want to `tag` the phase such that results are output to a unique
        directory whose names makes it explicit how the `Dataset` was changed.

        If this setting is off, the tag is an empty string and thus the directory structure is not changed.
        """

        super().__init__(search=search)

        self.settings = settings
        self.profiles = profiles

    def run(self, dataset: Dataset, mask):
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

        # This modifies the search path using the tag before the phase is run.
        self.modify_search_paths()

        analysis = self.make_analysis(dataset=dataset, mask=mask)

        result = self.run_analysis(analysis=analysis)

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

        """
        Here, the `SettingsPhase` are used to create the `MaskedDataset` that is fitted. 
        
        If the data_trim_left and / or data_trim_right settings are passed into the `SettingsPhase`, the function 
        below uses them to alter the `MaskedDataset`.

        Checkout `dataset/dataset.py` for more details.
        """

        masked_dataset = MaskedDataset(
            dataset=dataset, mask=mask, settings=self.settings.settings_masked_dataset
        )

        return Analysis(
            masked_dataset=masked_dataset, image_path=self.search.paths.image_path
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
