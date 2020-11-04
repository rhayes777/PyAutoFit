import autofit as af
from howtofit.chapter_1_introduction.tutorial_4_source_code.src.dataset.dataset import (
    Dataset,
)
from howtofit.chapter_1_introduction.tutorial_4_source_code.src.phase.result import (
    Result,
)
from howtofit.chapter_1_introduction.tutorial_4_source_code.src.phase.analysis import (
    Analysis,
)

"""
The phase package combines a data-set, model and `NonLinearSearch`, allowing us to fit the `Dataset` with the model. It
essentially acts as the `meeting point` between the other packages in the project (dataset, fit, plot) and modules
in the phase package (phase.py, analysis.py, result.py).
"""


class Phase(af.AbstractPhase):

    """
    This tells the phase that the input parameter `gaussian` is a model component that is fitted for by the phase`s
    `NonLinearSearch`.

    In `analysis.py`, the function `fit` has an input parameter called `instance` which is the gaussian mapped from
    this model via a unit vector and the model priors (as described in tutorial 1).

    For your model-fitting problem, this will be replaced by the modules in your `model` package.
    """

    gaussian = af.PhaseProperty("gaussian")

    Result = Result  # Set the result to the Result class in `result.py`

    def __init__(
        self,
        *,  # <- Requires inputs to be given as keyword arguments.
        gaussian,  # <- The user inputs a model -> gaussian.py -> `Gaussian` class here.
        search,  # <- This specifies the default `NonLinearSearch` used by the phase.
    ):
        """
        A phase which fits a `Gaussian` model using a `NonLinearSearch`.

        Parameters
        ----------
        gaussian : model.gaussians.Gaussian
            The model component `Gaussian` class fitted by this phase.
        search: class
            The class of a non_linear search
        """
        super().__init__(search=search)
        self.gaussian = gaussian

    def run(self, dataset: Dataset):
        """ Pass a `Dataset` to the phase, running the phase and `NonLinearSearch`.

        Parameters
        ----------
        `Dataset` : `Dataset`.Dataset
            The `Dataset` fitted by the phase, which is specified in the module `dataset/dataset.py`

        Returns
        -------
        result: result.Result
            A result object comprising information on the `NonLinearSearch` and the maximum likelihood model.
        """

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

        result = self.run_analysis(analysis=analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset):
        """
        Returns an Analysis object, which uses the `Dataset` with functions to perform a fit.

        Parameters
        ----------
        `Dataset` : `Dataset`.Dataset
            The `Dataset` fitted by the phase, which is specified in the module `dataset/dataset.py`

        Returns
        -------
        analysis : Analysis
            An analysis object that the `NonLinearSearch` calls to determine the fit log_likelihood for a given model
            instance.
        """
        return Analysis(dataset=dataset)

    def make_result(self, result, analysis):
        return self.Result(
            samples=result.samples,
            previous_model=self.model,
            search=self.search,
            analysis=analysis,
        )
