import json

from autoconf.dictable import to_dict, from_dict
from autofit.mapper.mock.mock_model import MockClassx2Tuple
import autofit as af


def test_tuple():
    model = af.Model(MockClassx2Tuple)
    json.dumps(model.dict())


class AttributeNotInInit:
    def __init__(self):
        self.attribute = 1


def test_embedded():
    model = af.Collection(instance=AttributeNotInInit())

    print(to_dict(model))
    from_dict(to_dict(model))
