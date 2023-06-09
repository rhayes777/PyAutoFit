import pytest

import autofit as af


class MockRelationship:
    def __call__(self, x):
        return 1.0


@pytest.fixture(name="path_relationship_map")
def make_path_relationship_map(interpolator):
    return {interpolator.gaussian.centre: af.Model(MockRelationship)}


def test(interpolator, path_relationship_map):
    interpolated = interpolator.get(
        interpolator.t == 1.0,
        path_relationship_map,
    )
    assert interpolated.gaussian.centre == 1.0


def test_model(interpolator, path_relationship_map):
    collection = interpolator.model(path_relationship_map)

    assert MockRelationship in [model.cls for model in collection]


def test_path_matching(path_relationship_map):
    assert ("gaussian", "centre") in path_relationship_map
