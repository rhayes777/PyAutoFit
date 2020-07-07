import autofit as af
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.dataset.dataset import (
    Dataset,
)
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.phase.result import (
    Result,
)
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.phase.analysis import (
    Analysis,
)
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.phase.meta_dataset import (
    MetaDataset,
)
from howtofit.chapter_1_introduction.tutorial_7_phase_customization.src.phase.settings import (
    PhaseSettings,
)


"""
The phase module has new features not included in tutorial 6, which customize the dataset that is fitted and tag
the output path of the results.
"""


class Phase(af.AbstractPhase):

    """
    Because we now have multiple profiles in our model, we have renamed 'gaussian' to 'profiles'. As before,
    PyAutoFit uses this information to map the input Profile classes to a model instance when performing a fit.
    """

    profiles = af.PhaseProperty("profiles")

    Result = Result

    @af.convert_paths
    def __init__(self, paths, profiles, settings, search):
        """
        A phase which fits a model composed of multiple profiles (Gaussian, Exponential) using a non-linear search.

        Parameters
        ----------
        paths : af.Paths
            Handles the output directory structure.
        profiles : [profiles.Profile]
            The model components (e.g. Gaussian, Exponenial) fitted by this phase.
        search: class
            The class of a non_linear search
        settings : PhaseSettings
            The collection of settings of the phase used to augment the data that is fitted and tag the output path.
        """

        """
        Here, we create a 'tag' for our phase. If we use an optional phase setting to alter the dataset we fit (here,
        a data_trim_ variable), we want to 'tag' the phase such that results are output to a unique
        directory whose names makes it explicit how the dataset was changed.

        If this setting is off, the tag is an empty string and thus the directory structure is not changed.
        """

        paths.tag = settings.tag  # The phase_tag must be manually added to the phase.

        super().__init__(paths=paths, search=search)

        self.profiles = profiles

        """
        Phase settings alter the dataset that is fitted, however a phase does not have access to the dataset until it
        is run (e.g. the run method below is passed the dataset). In order for a phase to use its input phase
        settings to create the dataset it fits, these settings are stored in the 'meta_dataset' attribute and used
        when the 'run' and 'make_analysis' methods are called.
        """

        self.meta_dataset = MetaDataset(settings=settings)

    def run(self, dataset: Dataset, mask):
        """
        Pass a dataset to the phase, running the phase and non-linear search.

        Parameters
        ----------
        dataset: aa.Dataset
            The dataset fitted by the phase, as defined in the 'dataset.py' module.
        mask: Mask
            The mask used for the analysis.

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising information on the non-linear search and the maximum likelihood model.
        """

        analysis = self.make_analysis(dataset=dataset, mask=mask)

        result = self.run_analysis(analysis=analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, mask):
        """
        Create an Analysis object, which creates the dataset and contains the functions which perform the fit.

        Parameters
        ----------
        dataset: aa.Dataset
            The dataset fitted by the phase, as defined in the 'dataset.py' module.

        Returns
        -------
        analysis : Analysis
            An analysis object that the non-linear search calls to determine the fit log_likelihood for a given model
            instance.
        """

        """
        Here, the meta_dataset is used to create the masked dataset that is fitted. If the data_trim_left and / or
        data_trim_right settings are passed into the phase, the function below uses them to alter the masked dataset.

        Checkout 'meta_dataset.py' for more details.
        """

        masked_dataset = self.meta_dataset.masked_dataset_from_dataset_and_mask(
            dataset=dataset, mask=mask
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
