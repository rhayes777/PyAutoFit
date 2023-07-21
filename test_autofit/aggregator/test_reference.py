from autofit.aggregator import Aggregator
from pathlib import Path
import autofit as af


def test_reference():
    aggregator = Aggregator(Path(__file__).parent)
    model = list(aggregator)[0].model
    assert model.cls is af.Gaussian
