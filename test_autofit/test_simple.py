import pytest

import autofit as af


class Class:
    def __init__(self, argument):
        self.argument = argument


def fitness_function(instance: Class):
    return -(instance.argument - 30) ** 2


@pytest.mark.parametrize(
    "optimizer",
    [
        af.MultiNest
    ]
)
def test_simple(optimizer):
    model = af.Model(
        Class,
        argument=af.UniformPrior(10, 100)
    )
    result = optimizer.simple_fit(
        model,
        fitness_function
    )

    assert result.instance.argument == pytest.approx(30, abs=1.0)
    assert result.likelihood == pytest.approx(0, abs=1)
