from autoconf.class_path import get_class_path
from autofit.aggregator import Aggregator
from pathlib import Path
import autofit as af


def test_without():
    aggregator = Aggregator(Path(__file__).parent)
    model = list(aggregator)[0].model
    assert model.cls is af.Gaussian


def test_with():
    aggregator = Aggregator(
        Path(__file__).parent, reference={"": get_class_path(af.Exponential)}
    )
    model = list(aggregator)[0].model
    assert model.cls is af.Exponential
