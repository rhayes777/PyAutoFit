import autofit as af


def test_gaussian_prior():
    prior = af.GaussianPrior(mean=1.0, sigma=1.0)

    children, aux_data = prior._tree_flatten()
    new = prior._tree_unflatten(aux_data, children)

    assert new.mean == prior.mean
    assert new.sigma == prior.sigma
    assert new.id == prior.id
