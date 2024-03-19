import autofit as af
from autofit.mapper.prior_model.representative import find_groups


def test_find_groups():
    info_paths = [((i, 1), 2) for i in range(3)]
    groups = find_groups(info_paths)
    assert groups == [(("0 - 2", 1), 2)]


def test_two_groups():
    info_paths = [((i, 1), 2) for i in range(3)] + [((i, 2), 2) for i in range(3)]
    groups = find_groups(info_paths)
    assert groups == [(("0 - 2", "1 - 2"), 2)]


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


def test_exception():
    info_paths = [((i, 0), 1) for i in range(3)] + [
        ((0, 1), 2),
        ((1, 1), 3),
        ((2, 1), 4),
    ]
    groups = find_groups(info_paths)
    assert groups == [(("0 - 2", 0), 1), ((0, 1), 2), ((1, 1), 3), ((2, 1), 4)]


def test_prefix():
    info_paths = [((0, i, 1), 2) for i in range(3)]
    groups = find_groups(info_paths)
    assert groups == [((0, "0 - 2", 1), 2)]
