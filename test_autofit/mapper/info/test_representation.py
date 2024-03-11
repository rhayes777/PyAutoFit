import autofit as af


def test_representative():
    collection = af.Collection([af.Model(af.Gaussian) for _ in range(10)])
    representative = af.Representative(collection)

    assert representative.prior_count == 3
