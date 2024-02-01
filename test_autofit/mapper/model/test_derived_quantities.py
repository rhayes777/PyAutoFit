import pytest

import autofit as af
from autofit import DirectoryPaths, DatabasePaths, SamplesPDF
from autofit.text.samples_text import derived_quantity_summary
from autofit.text.text_util import derived_info_from


pass


# def test_derived_quantities():
#     gaussian = af.Gaussian()
#
#     assert gaussian.upper_bound == 0.05
#
#     gaussian.upper_bound = 0.1
#     assert gaussian.upper_bound == 0.1
#
#
# def test_model_derived_quantities(model):
#     assert set(model.derived_quantities) == {
#         ("upper_bound",),
#         ("lower_bound",),
#     }
#
#
# def test_embedded_derived_quantities():
#     collection = af.Collection(
#         one=af.Gaussian,
#         two=af.Gaussian,
#     )
#
#     assert set(collection.derived_quantities) == {
#         ("one", "upper_bound"),
#         ("one", "lower_bound"),
#         ("two", "upper_bound"),
#         ("two", "lower_bound"),
#     }
#
#
# def test_multiple_levels():
#     collection = af.Collection(
#         one=af.Gaussian,
#         two=af.Collection(
#             three=af.Gaussian,
#         ),
#     )
#
#     assert set(collection.derived_quantities) == {
#         ("one", "upper_bound"),
#         ("one", "lower_bound"),
#         ("two", "three", "upper_bound"),
#         ("two", "three", "lower_bound"),
#     }
#
#
# @pytest.fixture(name="model")
# def make_model():
#     return af.Model(af.Gaussian)
#
#
# @pytest.fixture(name="samples")
# def make_samples(model):
#     return SamplesPDF(
#         model=model,
#         sample_list=[
#             af.Sample(
#                 log_likelihood=1.0,
#                 log_prior=2.0,
#                 weight=3.0,
#                 kwargs={
#                     "centre": 0.0,
#                     "normalization": 1.0,
#                     "sigma": 1.0,
#                 },
#             ),
#         ],
#     )
#
#
# def test_samples(samples):
#     derived_quantities = samples.derived_quantities_list[0]
#     assert derived_quantities == [-5.0, 5.0]
#
#
# def test_persist(samples, model):
#     paths = DirectoryPaths()
#     paths.model = model
#     paths.save_derived_quantities(samples)
#     assert paths._derived_quantities_file.exists()
#
#
# def test_persist_database(samples, model, session):
#     paths = DatabasePaths(session)
#     paths.model = model
#     paths.save_derived_quantities(samples)
#
#     assert paths.fit["derived_quantities"].shape == (1, 2)
#
#
# def test_summary(samples):
#     assert (
#         derived_quantity_summary(samples, median_pdf_model=False)
#         == """
#
# Summary (3.0 sigma limits):
#
# lower_bound        -5.0000 (-5.0000, -5.0000)
# upper_bound        5.0000 (5.0000, 5.0000)"""
#     )
#
#
# def test_derived_info_from(samples):
#     assert (
#         derived_info_from(samples)
#         == """Maximum Log Likelihood Model:
#
# lower_bound                                                                     -5.000
# upper_bound                                                                     5.000
#
#  WARNING: The samples have not converged enough to compute a PDF and model errors.
#  The model below over estimates errors.
#
#
#
# Summary (1.0 sigma limits):
#
# lower_bound                                                                     -5.0000 (-5.0000, -5.0000)
# upper_bound                                                                     5.0000 (5.0000, 5.0000)"""
#     )
#
#
# def test_derived_quantities_summary_dict(samples):
#     assert samples.derived_quantities_summary_dict == {
#         "max_log_likelihood_sample": {
#             "lower_bound": -5.0,
#             "upper_bound": 5.0,
#         },
#     }
