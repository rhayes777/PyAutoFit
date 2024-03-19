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

    from_dict(to_dict(model))


def test_serialise_compound():
    alpha = af.GaussianPrior(mean=1, sigma=0.03)
    gamma = 0.0
    beta = 1.0 + gamma - 2 * alpha

    assert beta.instance_from_prior_medians() == -1

    assert from_dict(to_dict(beta)).instance_from_prior_medians() == -1
