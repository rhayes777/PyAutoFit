import autofit as af


def test_has():
    model_object = af.ModelObject()
    assert not model_object.has(af.Gaussian)

    model_object.gaussian = af.Gaussian()
    assert model_object.has(af.Gaussian)

    assert model_object.has((af.Gaussian, str))
