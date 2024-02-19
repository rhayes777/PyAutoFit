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


def test_samples(samples):
    derived_quantities = samples.derived_quantities_list[0]
    assert derived_quantities == [2.3548200450309493]


def test_persist(samples, model):
    paths = DirectoryPaths()
    paths.model = model
    paths.save_derived_quantities(samples)
    assert paths._derived_quantities_file.exists()


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


@pytest.fixture(name="custom_model")
def make_custom_model():
    return af.Model(
        af.Gaussian,
        custom_derived_quantities={
            "custom": lambda instance: 2 * instance.centre,
        },
    )


def test_custom_derived_quantity(custom_model):
    instance = custom_model.instance_from_prior_medians()
    assert instance.centre == 0.5
    assert instance.custom == 1.0


def test_custom_derived_in_list(custom_model):
    assert set(custom_model.derived_quantities) == {
        ("fwhm",),
        ("custom",),
    }


def test_custom_derived_samples(samples, custom_model):
    samples.model = custom_model
    derived_quantities = samples.derived_quantities_list[0]
    assert derived_quantities == [2.3548200450309493, 0.0]


def test_custom_derived_summary(samples, custom_model):
    samples.model = custom_model
    assert (
        derived_quantity_summary(samples, median_pdf_model=False)
        == """

Summary (3.0 sigma limits):

fwhm          2.3548 (2.3548, 2.3548)
custom        0.0000 (0.0000, 0.0000)"""
    )
