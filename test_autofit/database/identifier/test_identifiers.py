import autofit as af
from autofit import database as db
from autofit.mock.mock import Gaussian


def test_fit():
    assert db.Fit(
        info={"info": 1}
    ).id != db.Fit(
        info={"info": 2}
    ).id


def test_prior():
    identifier = af.UniformPrior().identifier
    assert identifier == af.UniformPrior().identifier
    assert identifier != af.UniformPrior(
        lower_limit=0.5
    ).identifier
    assert identifier != af.UniformPrior(
        upper_limit=0.5
    ).identifier


def test_model():
    identifier = af.PriorModel(
        Gaussian,
        centre=af.UniformPrior()
    ).identifier
    assert identifier == af.PriorModel(
        Gaussian,
        centre=af.UniformPrior()
    ).identifier
    assert identifier != af.PriorModel(
        Gaussian,
        centre=af.UniformPrior(
            upper_limit=0.5
        )
    ).identifier


def test_collection():
    identifier = af.CollectionPriorModel(
        gaussian=af.PriorModel(
            Gaussian,
            centre=af.UniformPrior()
        )
    ).identifier
    assert identifier == af.CollectionPriorModel(
        gaussian=af.PriorModel(
            Gaussian,
            centre=af.UniformPrior()
        )
    ).identifier
    assert identifier != af.CollectionPriorModel(
        gaussian=af.PriorModel(
            Gaussian,
            centre=af.UniformPrior(
                upper_limit=0.5
            )
        )
    ).identifier


def test_instance():
    identifier = af.CollectionPriorModel(
        gaussian=Gaussian()
    ).identifier
    assert identifier == af.CollectionPriorModel(
        gaussian=Gaussian()
    ).identifier
    assert identifier != af.CollectionPriorModel(
        gaussian=Gaussian(
            centre=0.5
        )
    ).identifier
