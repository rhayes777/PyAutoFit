from typing import List

from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.samples.sample import Sample
from autofit.mapper.prior_model.collection import Collection
from autofit.mapper.prior.gaussian import GaussianPrior


class LatentSamples(Samples):
    def __init__(self, sample_list: List[Sample]):
        kwargs = sample_list[0].kwargs
        model = Collection()
        for key, value in kwargs.items():
            model[key] = GaussianPrior(
                mean=value,
                sigma=0.0,
            )
        super().__init__(model, sample_list)
