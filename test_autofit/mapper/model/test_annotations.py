from typing import Optional

import autofit as af
from autofit.mock.mock import Gaussian


class OptionalClass:
    def __init__(self, optional: Optional[Gaussian]):
        self.optional = optional


def test_optional():
    model = af.PriorModel(
        OptionalClass
    )

    assert model.instance_from_prior_medians().optional is None
