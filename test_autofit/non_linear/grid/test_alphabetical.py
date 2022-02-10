import pytest

import autofit as af


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.UniformPrior()


@pytest.fixture(
    name="model"
)
def make_model():
    model = af.Model(
        af.Gaussian
    )
    model.centre = af.UniformPrior()
    return model


@pytest.fixture(
    name="collection"
)
def make_collection(
        prior,
        model
):
    return af.Collection([
        prior,
        model
    ])


def test_jump_id():
    prior = af.UniformPrior()
    latest_id = af.UniformPrior().id

    prior.jump_id()

    assert prior.id == latest_id + 1


def test_alphabetise_model(model):
    assert model.priors_ordered_by_id[0] is not model.centre

    model.alphabetise()
    assert model.priors_ordered_by_id[0] is model.centre


def test_alphabetise_collection(
        collection,
        prior,
        model,
):
    collection.alphabetise()

    assert collection[0] == prior
    assert collection[1] == model

    assert collection[1].priors_ordered_by_id[0] is model.centre
