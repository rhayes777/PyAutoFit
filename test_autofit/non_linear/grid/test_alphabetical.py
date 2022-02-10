import autofit as af


def test_jump_id():
    prior = af.UniformPrior()
    latest_id = af.UniformPrior().id

    prior.jump_id()

    assert prior.id == latest_id + 1


def test_alphabetise_model():
    model = af.Model(
        af.Gaussian
    )
    model.centre = af.UniformPrior()

    assert model.priors_ordered_by_id[0] is not model.centre

    model.alphabetise()
    assert model.priors_ordered_by_id[0] is model.centre
