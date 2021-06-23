import autofit as af


def test_parameterization():
    model = af.Collection(
        collection=af.Collection(
            gaussian=af.Model(af.Gaussian)
        )
    )

    parameterization = model.parameterization
    assert parameterization == (
        'collection->gaussian:                                                       Gaussian (N=3)'
    )


def test_root():
    model = af.Model(af.Gaussian)
    parameterization = model.parameterization
    assert parameterization == (
        '(root):                                                                     Gaussian (N=3)'
    )
