import autofit as af


def test_duplication():
    model = af.Model(af.Gaussian)

    model.centre = model.sigma

    assert model.prior_count == 2

    model_dict = model.dict()
    new_model = af.Model.from_dict(model_dict)

    assert new_model.prior_count == 2
