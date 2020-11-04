from typing import List, Generator

from autofit import AbstractPriorModel, ModelInstance
from .non_linear.grid_search import make_lists


class Sensitivity:
    def __init__(
            self,
            perturbation_model: AbstractPriorModel,
            step_size=0.1
    ):
        self.step_size = step_size
        self.perturbation_model = perturbation_model

    @property
    def lists(self) -> List[List[float]]:
        return make_lists(
            self.perturbation_model.prior_count,
            step_size=self.step_size
        )

    @property
    def perturbation_instances(self) -> Generator[ModelInstance, None, None]:
        for list_ in self.lists:
            yield self.perturbation_model.instance_from_unit_vector(
                list_
            )
