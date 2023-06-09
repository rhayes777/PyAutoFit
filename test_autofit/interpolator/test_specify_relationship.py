import pytest

import autofit as af


class MockRelationship:
    def __call__(self, x):
        return 1.0


@pytest.fixture(name="path_relationship_map")
def make_path_relationship_map(interpolator):
    return {interpolator.gaussian.centre: af.Model(MockRelationship)}


@pytest.mark.parametrize("t", range(3))
def test(interpolator, path_relationship_map, t):
    interpolated = interpolator.get(
        interpolator.t == t,
        path_relationship_map,
    )
    assert interpolated.gaussian.centre == 1.0


def test_model(interpolator, path_relationship_map):
    collection = interpolator.model(path_relationship_map)

    assert MockRelationship in [model.cls for model in collection]


def test_path_matching(path_relationship_map):
    assert ("gaussian", "centre") in path_relationship_map
