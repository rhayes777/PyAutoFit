import autofit as af
from autofit import Samples


def test_derived_quantities():
    gaussian = af.Gaussian()

    assert gaussian.upper_bound == 0.05

    gaussian.upper_bound = 0.1
    assert gaussian.upper_bound == 0.1


def test_model_derived_quantities():
    model = af.Model(af.Gaussian)

    assert set(model.derived_quantities) == {
        ("upper_bound",),
        ("lower_bound",),
    }


def test_embedded_derived_quantities():
    collection = af.Collection(
        one=af.Gaussian,
        two=af.Gaussian,
    )

    assert set(collection.derived_quantities) == {
        ("one", "upper_bound"),
        ("one", "lower_bound"),
        ("two", "upper_bound"),
        ("two", "lower_bound"),
    }


def test_multiple_levels():
    collection = af.Collection(
        one=af.Gaussian,
        two=af.Collection(
            three=af.Gaussian,
        ),
    )

    assert set(collection.derived_quantities) == {
        ("one", "upper_bound"),
        ("one", "lower_bound"),
        ("two", "three", "upper_bound"),
        ("two", "three", "lower_bound"),
    }


def test_samples():
    samples = Samples(
        model=af.Model(af.Gaussian),
        sample_list=[
            af.Sample(
                log_likelihood=1.0,
                log_prior=2.0,
                weight=3.0,
                kwargs={
                    "centre": 0.0,
                    "normalization": 1.0,
                    "sigma": 1.0,
                },
            ),
        ],
    )
    derived_quantities = samples.derived_quantities_list[0]
    assert derived_quantities == [-5.0, 5.0]
