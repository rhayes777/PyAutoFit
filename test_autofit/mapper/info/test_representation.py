import autofit as af
from autofit.mapper.prior_model.representative import find_groups


def test_find_groups():
    info_paths = [((i, 1), 2) for i in range(3)]
    groups = find_groups(info_paths)
    assert groups == [(("0 - 2", 1), 2)]


def test_two_groups():
    info_paths = [((i, 1), 2) for i in range(3)] + [((i, 2), 2) for i in range(3)]
    groups = find_groups(info_paths)
    assert groups == [(("0 - 2", 1), 2), (("0 - 2", 2), 2)]


def test_equal_prior_values():
    prior = af.UniformPrior(0.0, 1.0)
    info_paths = [((i, 1), prior) for i in range(3)]
    groups = find_groups(info_paths)
    assert len(groups) == 1
    ((path, _),) = groups
    assert path == ("0 - 2", 1)


def test_different_priors():
    info_paths = [((i, 1), af.UniformPrior(0.0, i + 1)) for i in range(3)]
    groups = find_groups(info_paths)
    assert len(groups) == 3


def test_exception():
    info_paths = [((i, 0), 1) for i in range(3)] + [
        ((0, 1), 2),
        ((1, 1), 3),
        ((2, 1), 4),
    ]
    groups = find_groups(info_paths)
    assert groups == [(("0 - 2", 0), 1), (("0", 1), 2), (("1", 1), 3), (("2", 1), 4)]


def test_prefix():
    info_paths = [((0, i, 1), 2) for i in range(3)]
    groups = find_groups(info_paths)
    assert groups == [(("0", "0 - 2", 1), 2)]


def test_named_numeric_ranges():
    for prefix, count in (("lens_", 30), ("source_", 20)):
        paths = [((f"{prefix}{i}", "centre"), 1) for i in reversed(range(count))]
        assert find_groups(paths) == [
            ((f"{prefix}0 - {prefix}{count - 1}", "centre"), 1)
        ]


def test_named_range_fallbacks():
    for names, expected in [
        (["beta", "alpha"], "alpha - beta"),
        (["lens_9", "source_20"], "lens_9 - source_20"),
        (["lens_9", "lens_9"], "lens_9"),
        (["lens_02", "lens_2", "lens_10"], "lens_02 - lens_10"),
        (["1", "2", "4"], "1 - 2, 4"),
    ]:
        assert find_groups([((name, "centre"), 1) for name in names]) == [
            ((expected, "centre"), 1)
        ]
