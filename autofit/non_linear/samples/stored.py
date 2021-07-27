from typing import List, Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples.sample import Sample
from .pdf import PDFSamples


class StoredSamples(PDFSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            sample_list: List[Sample],
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
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
            time=time,
        )

        self._unconverged_sample_size = int(unconverged_sample_size)
