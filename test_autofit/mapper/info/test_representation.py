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
    info_paths = [((i, 1), 2) for i in range(3)]
    groups = find_groups(info_paths)
    assert groups == [(("0 - 2", 1), 2)]


def test_two_groups():
    info_paths = [((i, 1), 2) for i in range(3)] + [((i, 2), 2) for i in range(3)]
    groups = find_groups(info_paths)
    assert groups == [(("0 - 2", 1), 2), (("0 - 2", 2), 2)]


def test_equal_prior_values():
    info_paths = [((i, 1), af.UniformPrior(0.0, 1.0)) for i in range(3)]
    groups = find_groups(info_paths)
    assert len(groups) == 1
    ((path, _),) = groups
    assert path == ("0 - 2", 1)


def test_different_priors():
    info_paths = [((i, 1), af.UniformPrior(0.0, i + 1)) for i in range(3)]
    groups = find_groups(info_paths)
    assert len(groups) == 3
