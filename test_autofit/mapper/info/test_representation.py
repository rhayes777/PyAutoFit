import autofit as af
from autofit.mapper.prior_model.representative import find_groups


def test_simple_keys():
    collection = af.Collection([af.Model(af.Gaussian) for _ in range(10)])
    representative = af.Representative(collection)

    keys, _ = zip(*representative.info_tuples)
    assert keys == (
        ("0 - 9", "centre"),
        ("0 - 9", "normalization"),
        ("0 - 9", "sigma"),
    )


def test_different_centres():
    collection = af.Collection(
        [
            af.Model(
                af.Gaussian,
                centre=i,
            )
            for i in range(10)
        ]
    )
    representative = af.Representative(collection)

    keys, _ = zip(*representative.info_tuples)
    assert keys == (
        ("0 - 9", "centre"),
        ("0 - 9", "normalization"),
        ("0 - 9", "sigma"),
    ) + tuple((str(i), "centre") for i in range(10))


def test_find_groups():
    info_paths = [
        (
            (1, 2),
            1,
        ),
        (
            (2, 2),
            1,
        ),
        (
            (3, 2),
            1,
        ),
    ]
    groups = find_groups(info_paths)
    assert groups == [(("1 - 3", 2), 1)]
