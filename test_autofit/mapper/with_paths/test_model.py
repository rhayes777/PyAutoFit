import pytest

import autofit as af
from autofit.mapper.prior_model.abstract import paths_to_tree


class TestWithoutAttributes:
    @pytest.mark.parametrize(
        "attribute",
        [
            "centre",
            "normalization",
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


class TestWith:
    def test_subpath(self, model):
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

    def test_path(self, model):
        with_paths = model.with_paths([
            ("gaussian_1", "centre"),
            ("gaussian_2", "normalization"),
        ])

        gaussian_1 = with_paths.gaussian_1
        assert hasattr(gaussian_1, "centre")
        assert not hasattr(gaussian_1, "normalization")

        gaussian_2 = with_paths.gaussian_2
        assert not hasattr(gaussian_2, "centre")
        assert hasattr(gaussian_2, "normalization")


def test_string_paths(model):
    with_paths = model.with_paths([
        "gaussian_1.centre",
        "gaussian_2.normalization",
    ])

    gaussian_1 = with_paths.gaussian_1
    assert hasattr(gaussian_1, "centre")
    assert not hasattr(gaussian_1, "normalization")

    gaussian_2 = with_paths.gaussian_2
    assert not hasattr(gaussian_2, "centre")
    assert hasattr(gaussian_2, "normalization")


class TestWithout:
    def test_subpath(self, model):
        with_paths = model.without_paths([
            ("gaussian_1",)
        ])
        assert not hasattr(
            with_paths,
            "gaussian_1"
        )
        assert hasattr(
            with_paths.gaussian_2,
            "centre"
        )

    def test_path(self, model):
        with_paths = model.without_paths([
            ("gaussian_1", "centre"),
            ("gaussian_2", "normalization"),
        ])

        gaussian_1 = with_paths.gaussian_1
        assert not hasattr(gaussian_1, "centre")
        assert hasattr(gaussian_1, "normalization")

        gaussian_2 = with_paths.gaussian_2
        assert hasattr(gaussian_2, "centre")
        assert not hasattr(gaussian_2, "normalization")


class TestPathsToTree:
    def test_trivial(self):
        assert paths_to_tree([
            ("one", "two")
        ]) == {
                   "one": {
                       "two": {}
                   }
               }

    def test_multiple(self):
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

    def test_second_tier(self):
        assert paths_to_tree([
            ("one", "two", "three"),
            ("one", "two", "four"),
        ]) == {
                   "one": {
                       "two": {
                           "three": {},
                           "four": {}
                       }
                   }
               }


@pytest.mark.parametrize(
    'path, index',
    [
        (("gaussian_1", "centre"), 0),
        (("gaussian_2", "centre"), 3),
        (("gaussian_2", "normalization"), 4),
    ]
)
def test_indices(
        model,
        path,
        index
):
    assert model.index(
        path
    ) == index


@pytest.fixture(
    name="tuple_model"
)
def make_tuple_model():
    return af.Model(af.m.MockWithTuple)


class TestTuples:
    def test_with_specific(self, tuple_model):
        with_paths = tuple_model.with_paths([
            ("tup", "0")
        ])

        assert hasattr(with_paths, "tup_0")
        assert not hasattr(with_paths, "tup_1")

    def test_without_specific(self, tuple_model):
        with_paths = tuple_model.without_paths([
            ("tup", "0")
        ])

        assert not hasattr(with_paths, "tup_0")
        assert hasattr(with_paths, "tup_1")

    def test_with(self, tuple_model):
        with_paths = tuple_model.with_paths([
            ("tup",)
        ])

        assert hasattr(with_paths, "tup")

    def test_without(self, tuple_model):
        with_paths = tuple_model.without_paths([
            ("tup",)
        ])

        assert not hasattr(with_paths, "tup")

    def test_with_explicit(self, tuple_model):
        with_paths = tuple_model.with_paths([
            ("tup", "tup_0")
        ])

        assert hasattr(with_paths, "tup_0")
        assert not hasattr(with_paths, "tup_1")

    def test_without_explicit(self, tuple_model):
        with_paths = tuple_model.without_paths([
            ("tup", "tup_0")
        ])

        assert not hasattr(with_paths, "tup_0")
        assert hasattr(with_paths, "tup_1")
