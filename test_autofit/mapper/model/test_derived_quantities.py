from autofit import Gaussian


def test_derived_quantities():
    gaussian = Gaussian()

    assert gaussian.upper_bound == 0.05

    gaussian.upper_bound = 0.1
    assert gaussian.upper_bound == 0.1
