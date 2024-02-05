import pytest

import autofit as af
from autofit import DirectoryPaths, DatabasePaths, SamplesPDF
from autofit.text.samples_text import derived_quantity_summary
from autofit.text.text_util import derived_info_from


def test_derived_quantities():
    gaussian = af.Gaussian()

    assert gaussian.fwhm == 0.023548200450309493


def test_model_derived_quantities(model):
    assert set(model.derived_quantities) == {
        ("fwhm",),
    }


def test_embedded_derived_quantities():
    collection = af.Collection(
        one=af.Gaussian,
        two=af.Gaussian,
    )

    assert set(collection.derived_quantities) == {
        ("one", "fwhm"),
        ("two", "fwhm"),
    }


def test_multiple_levels():
    collection = af.Collection(
        one=af.Gaussian,
        two=af.Collection(
            three=af.Gaussian,
        ),
    )

    assert set(collection.derived_quantities) == {
        ("one", "fwhm"),
        ("two", "three", "fwhm"),
    }


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.Gaussian)


@pytest.fixture(name="samples")
def make_samples(model):
    return SamplesPDF(
        model=model,
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


def test_samples(samples):
    derived_quantities = samples.derived_quantities_list[0]
    assert derived_quantities == [2.3548200450309493]


def test_persist(samples, model):
    paths = DirectoryPaths()
    paths.model = model
    paths.save_derived_quantities(samples)
    assert paths._derived_quantities_file.exists()


def test_persist_database(samples, model, session):
    paths = DatabasePaths(session)
    paths.model = model
    paths.save_derived_quantities(samples)

    assert paths.fit["derived_quantities"].shape == (1, 1)


def test_summary(samples):
    assert (
        derived_quantity_summary(samples, median_pdf_model=False)
        == """

Summary (3.0 sigma limits):

fwhm        2.3548 (2.3548, 2.3548)"""
    )


def test_derived_info_from(samples):
    assert (
        derived_info_from(samples)
        == """Maximum Log Likelihood Model:

fwhm                                                                            2.355

 WARNING: The samples have not converged enough to compute a PDF and model errors. 
 The model below over estimates errors. 



Summary (1.0 sigma limits):

fwhm                                                                            2.3548 (2.3548, 2.3548)"""
    )


def test_derived_quantities_summary_dict(samples):
    assert samples.derived_quantities_summary_dict == {
        "max_log_likelihood_sample": {
            "fwhm": 2.3548200450309493,
        },
    }


def test_custom_derived_quantity():
    model = af.Model(
        af.Gaussian,
        custom_derived_quantities={
            "custom": lambda instance: 1.0,
        },
    )
    instance = model.instance_from_prior_medians()
    assert instance.custom == 1.0
