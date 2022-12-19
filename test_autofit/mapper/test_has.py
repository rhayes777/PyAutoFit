import autofit as af

class GaussianChild(af.Gaussian):
    pass


def test_inheritance():
    collection = af.Collection(
        first=af.Model(
            GaussianChild
        ),
        second=GaussianChild()
    )

    assert collection.has_instance(
        af.Gaussian
    )
    assert collection.has_model(
        af.Gaussian
    )


def test_embedded():
    collection = af.Collection(
        model=af.Model(
            af.Gaussian,
            centre=GaussianChild
        )
    )
    assert collection.has_model(
        GaussianChild
    )


def test_no_free_parameters():
    collection = af.Collection(
        gaussian=af.Model(
            af.Gaussian,
            centre=1.0,
            normalization=1.0,
            sigma=1.0,
        )
    )
    assert collection.prior_count == 0
    assert collection.has_model(
        af.Gaussian
    ) is False


def test_instance():
    collection = af.Collection(
        gaussian=af.Gaussian()
    )

    assert collection.has_instance(
        af.Gaussian
    ) is True
    assert collection.has_model(
        af.Gaussian
    ) is False


def test_model():
    collection = af.Collection(
        gaussian=af.Model(
            af.Gaussian
        )
    )

    assert collection.has_model(
        af.Gaussian
    ) is True
    assert collection.has_instance(
        af.Gaussian
    ) is False


def test_both():
    collection = af.Collection(
        gaussian=af.Model(
            af.Gaussian
        ),
        gaussian_2=af.Gaussian()
    )

    assert collection.has_model(
        af.Gaussian
    ) is True
    assert collection.has_instance(
        af.Gaussian
    ) is True


def test_embedded():
    collection = af.Collection(
        gaussian=af.Model(
            af.Gaussian,
            centre=af.Gaussian()
        ),
    )

    assert collection.has_model(
        af.Gaussian
    ) is True
    assert collection.has_instance(
        af.Gaussian
    ) is True


def test_is_only_model():
    collection = af.Collection(
        gaussian=af.Model(
            af.Gaussian
        ),
        gaussian_2=af.Model(
            af.Gaussian
        )
    )

    assert collection.is_only_model(
        af.Gaussian
    ) is True

    collection.other = af.Model(
        af.m.MockClassx2
    )

    assert collection.is_only_model(
        af.Gaussian
    ) is False
