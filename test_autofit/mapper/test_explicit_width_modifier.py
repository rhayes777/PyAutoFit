import pytest

import autofit as af


@pytest.fixture(name="width_modifier")
def make_width_modifier():
    return af.RelativeWidthModifier(2.0)


@pytest.fixture(name="updated_model")
def make_updated_model(width_modifier):
    model = af.Model(af.Gaussian)
    model.centre.width_modifier = width_modifier

    return model.mapper_from_gaussian_tuples([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0),])


def test_explicit_width_modifier(updated_model):
    assert updated_model.centre.sigma == 2.0
    assert updated_model.normalization.sigma == 1.0


def test_propagation(updated_model, width_modifier):
    assert updated_model.centre.width_modifier is width_modifier
