import autofit as af
from autofit.mapper.identifier import Identifier


def test_gaussian_prior():
    assert str(
        Identifier(
            af.GaussianPrior(
                mean=1.0,
                sigma=2.0
            )
        )
    ) == "218e05b43472cb7661b4712da640a81c"


def test_uniform_prior():
    assert str(
        Identifier(
            af.UniformPrior(
                lower_limit=1.0,
                upper_limit=2.0
            )
        )
    ) == "a0de90b9099d70b945dc56094eb5c8de"


def test_logarithmic_prior():
    assert str(
        Identifier(
            af.LogUniformPrior(
                lower_limit=1.0,
                upper_limit=2.0
            )
        )
    ) == "0e8220c88678dcb31a398f9a34dcbc8a"


def test_model_identifier():
    assert af.Model(
        af.Gaussian
    ).identifier == "1719f29d2938d146d230d52ef7379a84"
