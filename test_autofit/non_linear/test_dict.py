import autofit as af
from autoconf.class_path import get_class_path


def to_dict(obj):
    get_class_path(obj)


def test_dict():
    dynesty = af.DynestyStatic()
    print(dynesty.dict())


def test_prior_passer():
    prior_passer = af.PriorPasser(
        sigma=1.0,
        use_errors=False,
        use_widths=False,
    )
    assert prior_passer.dict() == {
        "sigma": 1.0,
        "use_errors": False,
        "use_widths": False,
    }


def test_initializer():
    initializer = af.InitializerBall(lower_limit=0.0, upper_limit=1.0)
    assert initializer.dict() == {
        "lower_limit": 0.0,
        "upper_limit": 1.0,
    }
