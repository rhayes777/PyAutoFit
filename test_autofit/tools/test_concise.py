import autofit as af
import pytest

from autofit.text.representative import Representative


@pytest.fixture(name="collection")
def make_collection():
    return af.Collection([af.Model(af.Gaussian) for _ in range(20)])


def test_model_info(collection):
    print(collection.info)


def test_representative(collection):
    ((key, representative),) = Representative.find_representatives(collection.items())

    assert len(representative.children) == 20
    assert key == "0 - 9"


def test_get_blueprint():
    assert Representative.get_blueprint(af.Model(af.Gaussian)) == (
        (("centre",), (af.UniformPrior, "lower_limit = 0.0, upper_limit = 1.0")),
        (("normalization",), (af.UniformPrior, "lower_limit = 0.0, upper_limit = 1.0")),
        (("sigma",), (af.UniformPrior, "lower_limit = 0.0, upper_limit = 1.0")),
    )
