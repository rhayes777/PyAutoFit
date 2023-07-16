from typing import Dict, List, Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples.sample import Sample
from .pdf import SamplesPDF


class SamplesStored(SamplesPDF):

    def __init__(
            self,
            model: AbstractPriorModel,
            sample_list: List[Sample],
            samples_info : Optional[Dict] = None,
    ):
        """
        The `Samples` of a non-linear search, specifically the samples of a `NonLinearSearch` which maps out the
        posterior of parameter space and thus does provide information on parameter errors.

        Parameters
        ----------
        model : af.ModelMapper
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        super().__init__(
            model=model,
            sample_list=sample_list,
            samples_info=samples_info,
        )
