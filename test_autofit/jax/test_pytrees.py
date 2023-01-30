import autofit as af


def recreate(o):
    children, aux_data = o._tree_flatten()
    return type(o)._tree_unflatten(aux_data, children)


def test_gaussian_prior():
    prior = af.GaussianPrior(mean=1.0, sigma=1.0)

    new = recreate(prior)

    assert new.mean == prior.mean
    assert new.sigma == prior.sigma
    assert new.id == prior.id


def test_model():
    model = af.Model(
        af.Gaussian,
        centre=af.GaussianPrior(mean=1.0, sigma=1.0),
        normalization=af.GaussianPrior(mean=1.0, sigma=1.0, lower_limit=0.0),
        sigma=af.GaussianPrior(mean=1.0, sigma=1.0, lower_limit=0.0),
    )

    new = recreate(model)
    assert new.cls == af.Gaussian

    centre = new.centre
    assert centre.mean == model.centre.mean
    assert centre.sigma == model.centre.sigma
    assert centre.id == model.centre.id
