import json

from autofit.mapper.mock.mock_model import MockClassx2Tuple
import autofit as af


def test_tuple():
    model = af.Model(MockClassx2Tuple)
    json.dumps(model.dict())
