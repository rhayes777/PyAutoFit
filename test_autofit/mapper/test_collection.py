import autofit as af


def test_collection():
    collection = af.Collection(
        af.Model(af.Gaussian) for _ in range(10)
    )
    assert collection.prior_count == 30
