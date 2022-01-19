from typing import Optional

import autofit as af


class OptionalClass:
    def __init__(self, optional: Optional[af.Gaussian]):
        self.optional = optional


def test_optional():
    model = af.Model(
        OptionalClass
    )

    assert model.instance_from_prior_medians().optional is None


class DefaultClass:
    def __init__(self, default=1.0):
        self.default = default

    __default_fields__ = ("default",)


def test_default():
    model = af.Model(
        DefaultClass
    )
    assert model.instance_from_prior_medians().default == 1.0
