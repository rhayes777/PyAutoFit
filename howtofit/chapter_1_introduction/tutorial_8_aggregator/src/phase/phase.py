import autofit as af
from howtofit.chapter_1_introduction.tutorial_8_aggregator.src.dataset.dataset import (
    Dataset,
)
from howtofit.chapter_1_introduction.tutorial_8_aggregator.src.phase.result import (
    Result,
)
from howtofit.chapter_1_introduction.tutorial_8_aggregator.src.phase.analysis import (
    Analysis,
)
from howtofit.chapter_1_introduction.tutorial_8_aggregator.src.phase.meta_dataset import (
    MetaDataset,
)
from howtofit.chapter_1_introduction.tutorial_8_aggregator.src.phase.settings import (
    PhaseSettings,
)


# The 'phase.py' module is mostly unchanged from the previous tutorial, however the 'run' function has been updated.


class Phase(af.AbstractPhase):

    profiles = af.PhaseProperty("profiles")

    Result = Result

    @af.convert_paths
    def __init__(self, paths, *, profiles, settings, search):
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
        data_trim_left : int or None
            The number of pixels by which the data is trimmed from the left-hand side.
        data_trim_right : int or None
            The number of pixels by which the data is trimmed from the right-hand side.
        """

        paths.tag = settings.tag

        super().__init__(paths=paths, search=search)

        self.profiles = profiles

        self.meta_dataset = MetaDataset(settings=settings)

    @property
    def folders(self):
        return self.search.folders

    def run(self, dataset: Dataset, mask, info=None):
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

        # These functions save the objects we will later access using the aggregator. They are saved via the 'pickle'
        # module in Python, which serializes the data on to the hard-disk.

        # See the 'dataset.py' module for a description of what the metadata is.

        self.save_metadata(dataset=dataset)
        self.save_dataset(dataset=dataset)
        self.save_mask(mask=mask)
        self.save_meta_dataset(meta_dataset=self.meta_dataset)
        # This saves the search information of the phase, meaning that we can use the search instance
        # (e.g. Emcee) to interpret our results in the aggregator.

        analysis = self.make_analysis(dataset=dataset, mask=mask)

        result = self.run_analysis(analysis=analysis, info=info)

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
