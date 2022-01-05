import pytest

import autofit as af
from autofit.mapper.prior_model.abstract import paths_to_tree


@pytest.fixture(
    name="gaussian_1"
)
def make_gaussian_1():
    return af.Model(
        af.Gaussian
    )


@pytest.fixture(
    name="model"
)
def make_model(gaussian_1):
    return af.Collection(
        gaussian_1=gaussian_1,
        gaussian_2=af.Model(
            af.Gaussian
        ),
    )


class TestWithoutAttributes:
    @pytest.mark.parametrize(
        "attribute",
        [
            "centre",
            "intensity",
            "sigma",
        ]
    )
    def test_gaussian(
            self,
            gaussian_1,
            attribute
    ):
        without = gaussian_1.without_attributes()
        assert not hasattr(
            without,
            attribute
        )
        assert hasattr(
            without, "cls"
        )

    @pytest.mark.parametrize(
        "attribute",
        [
            "gaussian_1",
            "gaussian_2"
        ]
    )
    def test_collection(
            self,
            model,
            attribute
    ):
        without = model.without_attributes()
        assert not hasattr(
            without,
            attribute
        )


def test_with_paths(model):
    with_paths = model.with_paths([
        ("gaussian_1",)
    ])
    assert not hasattr(
        with_paths,
        "gaussian_2"
    )
    assert hasattr(
        with_paths.gaussian_1,
        "centre"
    )


def test_attributes(model):
    with_paths = model.with_paths([
        ("gaussian_1", "centre"),
        ("gaussian_2", "intensity"),
    ])

    gaussian_1 = with_paths.gaussian_1
    assert hasattr(gaussian_1, "centre")
    assert not hasattr(gaussian_1, "intensity")

    gaussian_2 = with_paths.gaussian_2
    assert not hasattr(gaussian_2, "centre")
    assert hasattr(gaussian_2, "intensity")


def test_paths_to_tree():
    assert paths_to_tree([
        ("one", "two")
    ]) == {
               "one": {
                   "two": {}
               }
           }
    assert paths_to_tree([
        ("one", "two"),
        ("one", "three"),
        ("two", "three"),
    ]) == {
               "one": {
                   "two": {},
                   "three": {},
               },
               "two": {
                   "three": {}
               }
           }
