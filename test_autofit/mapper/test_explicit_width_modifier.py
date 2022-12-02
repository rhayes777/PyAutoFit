import autofit as af


def test_explicit_width_modifier():
    model = af.Model(af.Gaussian)
    model.centre.width_modifier = af.RelativeWidthModifier(2.0)

    updated = model.mapper_from_gaussian_tuples([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0),])

    assert updated.centre.sigma == 2.0
    assert updated.normalization.sigma == 1.0
