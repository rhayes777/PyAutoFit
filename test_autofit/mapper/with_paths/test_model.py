import pytest

from autofit.mapper.prior_model.abstract import paths_to_tree


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
            ("gaussian_2", "intensity"),
        ])

        gaussian_1 = with_paths.gaussian_1
        assert hasattr(gaussian_1, "centre")
        assert not hasattr(gaussian_1, "intensity")

        gaussian_2 = with_paths.gaussian_2
        assert not hasattr(gaussian_2, "centre")
        assert hasattr(gaussian_2, "intensity")


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
            ("gaussian_2", "intensity"),
        ])

        gaussian_1 = with_paths.gaussian_1
        assert not hasattr(gaussian_1, "centre")
        assert hasattr(gaussian_1, "intensity")

        gaussian_2 = with_paths.gaussian_2
        assert hasattr(gaussian_2, "centre")
        assert not hasattr(gaussian_2, "intensity")


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


def test_indices(
        model,
):
    assert model.index(
        ("gaussian_1", "centre")
    ) == 0
