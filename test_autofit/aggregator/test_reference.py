import pytest

from autoconf.class_path import get_class_path
from autofit.aggregator import Aggregator
from pathlib import Path
import autofit as af


@pytest.fixture(name="directory")
def make_directory():
    return Path(__file__).parent


def test_without(directory):
    aggregator = Aggregator(directory)
    model = list(aggregator)[0].model
    assert model.cls is af.Gaussian


def test_with():
    aggregator = Aggregator(
        Path(__file__).parent, reference={"": get_class_path(af.Exponential)}
    )
    model = list(aggregator)[0].model
    assert model.cls is af.Exponential


def test_database(session, directory):
    aggregator = af.Aggregator(session)
    aggregator.add_directory(directory)

    session.commit()

    fit = list(aggregator)[0]
a    model = fit.model
    assert model.cls is af.Gaussian
