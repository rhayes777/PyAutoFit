import pytest

import autofit as af


@pytest.fixture(
    name="frozen_model"
)
def make_frozen_model():
    model = af.PriorModel(
        af.Gaussian
    )

    model.freeze()
    return model


@pytest.fixture(
    name="frozen_collection"
)
def make_frozen_collection():
    model = af.CollectionPriorModel(
        gaussian=af.PriorModel(
            af.Gaussian
        )
    )

    model.freeze()
    return model


def test_instantiate(
        frozen_model
):
    frozen_model.instance_from_prior_medians()


def test_modify(
        frozen_model
):
    with pytest.raises(
            AssertionError
    ):
        frozen_model.key = "value"


class TestFrozenCollection:
    def test_append(
            self,
            frozen_collection
    ):
        with pytest.raises(
                AssertionError
        ):
            frozen_collection.append(
                af.Gaussian()
            )

    def test_set_item(
            self,
            frozen_collection
    ):
        with pytest.raises(
                AssertionError
        ):
            frozen_collection["key"] = "value"

    def test_children_frozen(
            self,
            frozen_collection
    ):
        with pytest.raises(
                AssertionError
        ):
            frozen_collection[0].key = "value"


def test_unfreeze(
        frozen_collection
):
    frozen_collection.unfreeze()
    frozen_collection["key"] = "value"
    frozen_collection[0].key = "value"
