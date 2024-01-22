import autofit as af


def test_derived_quantities():
    gaussian = af.Gaussian()

    assert gaussian.upper_bound == 0.05

    gaussian.upper_bound = 0.1
    assert gaussian.upper_bound == 0.1


def test_model_derived_quantities():
    model = af.Model(af.Gaussian)

    assert len(model.derived_quantities) == 2


def test_embedded_derived_quantities():
    collection = af.Collection(
        one=af.Gaussian,
        two=af.Gaussian,
    )

    assert len(collection.derived_quantities) == 4
