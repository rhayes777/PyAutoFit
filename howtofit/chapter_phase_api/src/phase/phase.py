import autofit as af
from src.dataset.dataset import Dataset
from src.phase.result import Result
from src.phase.analysis import Analysis
from src.phase.settings import SettingsPhase

"""
The phase package combines a data-set, model and `NonLinearSearch`, allowing us to fit the `Dataset` with the model. It
essentially acts as the `meeting point` between the other packages in the project (dataset, fit, plot) and modules
in the phase package (phase.py, analysis.py, result.py).
"""


class Phase(af.AbstractPhase):

    """
    This tells the phase that the input parameter `profiles` contains model components that are fitted for by the
    phase`s `NonLinearSearch`.

    In `analysis.py`, the `log_likelihood_function`' input parameter `instance` contains the `profiles` mapped from
    this model via the `NonLinearSearch` (as we saw in chapter 1).

    For your model-fitting problem, this will be replaced by the modules in your `model` package.
    """

    profiles = af.PhaseProperty("profiles")

    Result = Result  # Set the result to the Result class in `result.py`

    def __init__(
        self,
        search: af.NonLinearSearch,  # <- This specifies the default `NonLinearSearch` used by the phase.
        settings: SettingsPhase,  # <- Settings will be covered in detail in tutorial 3.
        profiles: list,
    ):
        """
        A phase which fits a `Gaussian` model using a `NonLinearSearch`.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        settings : SettingsPhase
            The collection of settings of the phase used to augment the data that is fitted and tag the output path.
        profiles : [profiles.Profile]
            The model components (e.g. Gaussian, Exponential) fitted by this phase.
        """
        super().__init__(search=search)

        self.settings = settings
        self.profiles = profiles

    def run(self, dataset: Dataset, info=None) -> Result:
        """
        Pass a `Dataset` to the phase, running the phase and `NonLinearSearch`.

        Parameters
        ----------
        dataset : `Dataset`.Dataset
            The `Dataset` fitted by the phase, which is specified in the module `dataset/dataset.py`

        Returns
        -------
        result: result.Result
            A result object comprising information on the `NonLinearSearch` and the maximum likelihood model.
        """

        """Tutorial 3 will cover phase tagging, which this function handles."""

        self.modify_search_paths()

        """
        These functions create instances of the Analysis class (in `analysis.py`), runs the analysis (which performs
        the `NonLinearSearch` ) and returns an instance of the Result class (in `result.py`).

        Once you`ve looked through this module, check those modules out to see exactly what these classes do!
        """

        analysis = self.make_analysis(dataset=dataset)

        """
        `run_analysis` is not located in analysis.py, instead it is an inherited method from the parent class
        `af.AbstractPhase`. Essentially, all this function does is begin the `NonLinearSearch`, using the analysis
        created above.
        """

        result = self.run_analysis(analysis=analysis, info=info)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset: Dataset) -> Analysis:
        """
        Returns an Analysis object, which uses the `Dataset` with functions to perform a fit.

        Parameters
        ----------
        dataset : `Dataset`.Dataset
            The `Dataset` fitted by the phase, which is specified in the module `dataset/dataset.py`

        Returns
        -------
        analysis : Analysis
            An analysis object that the `NonLinearSearch` calls to determine the fit log_likelihood for a given model
            instance.
        """

        dataset = dataset.trimmed_dataset_from_settings(
            settings=self.settings.settings_dataset
        )

        return Analysis(dataset=dataset, settings=self.settings)

    def make_result(self, result: af.Result, analysis: Analysis) -> Result:
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
