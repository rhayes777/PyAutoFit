import autofit as af


def test_has():
    model_object = af.ModelObject()
    assert not model_object.has(af.ex.Gaussian)

    model_object.gaussian = af.ex.Gaussian()
    assert model_object.has(af.ex.Gaussian)

    assert model_object.has((af.ex.Gaussian, str))
