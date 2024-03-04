import itertools

import autofit as af
import pytest

from autofit.text.representative import Representative
from autofit.text.text_util import result_info_from
from autofit.visualise import VisualiseGraph


@pytest.fixture(autouse=True)
def reset_ids():
    af.Prior._ids = itertools.count()


@pytest.fixture(name="collection")
def make_collection():
    return af.Collection([af.Model(af.Gaussian) for _ in range(20)])


def test_model_info(collection):
    assert (
        collection.info
        == """Total Free Parameters = 60

model                                                                           Collection (N=60)
    0 - 9                                                                       Gaussian (N=3)

0 - 9
    centre                                                                      UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
    normalization                                                               UniformPrior [1], lower_limit = 0.0, upper_limit = 1.0
    sigma                                                                       UniformPrior [2], lower_limit = 0.0, upper_limit = 1.0"""
    )


def test_representative(collection):
    ((key, representative),) = Representative.find_representatives(collection.items())

    assert len(representative.children) == 20
    assert key == "0 - 9"


def test_reference_count(collection):
    assert collection.reference_count(collection[0].centre) == 1

    collection[5].centre = collection[0].centre
    assert collection.reference_count(collection[0].centre) == 2


def test_find_representatives(collection):
    assert len(Representative.find_representatives(collection.items())) == 1

    collection[0].centre = af.UniformPrior(0.0, 1.0)
    assert len(Representative.find_representatives(collection.items())) == 2


def test_mid_collection_anomaly(collection):
    collection[5].centre = af.UniformPrior(0.0, 1.0)
    assert len(Representative.find_representatives(collection.items())) == 3


def test_shared_prior(collection):
    collection[5].centre = collection[0].centre
    assert len(Representative.find_representatives(collection.items())) == 4


def test_shared_descendents(collection):
    assert Representative.shared_descendents(collection) == set()

    shared = collection[1].centre
    collection[0].centre = shared
    assert Representative.shared_descendents(collection) == {shared}


@pytest.fixture(name="samples")
def make_samples(collection):
    parameters = [len(collection) * [1.0, 2.0, 3.0]]

    log_likelihood_list = [1.0]

    return af.m.MockSamples(
        model=collection,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0],
            weight_list=log_likelihood_list,
            model=collection,
        ),
        max_log_likelihood_instance=collection.instance_from_prior_medians(),
    )


def test_model_results(samples):
    assert (
        result_info_from(samples)
        == """Maximum Log Likelihood                                                          1.00000000
Maximum Log Posterior                                                           1.00000000

model                                                                           Collection (N=60)
    0 - 9                                                                       Gaussian (N=3)

Maximum Log Likelihood Model:

0 - 9
    centre                                                                      1.000
    normalization                                                               2.000
    sigma                                                                       3.000

 WARNING: The samples have not converged enough to compute a PDF and model errors. 
 The model below over estimates errors. 



Summary (1.0 sigma limits):

0 - 9
    centre                                                                      1.00 (1.00, 1.00)
    normalization                                                               2.00 (2.00, 2.00)
    sigma                                                                       3.00 (3.00, 3.00)

instances

"""
    )


def test_visualise(collection, output_directory):
    collection[5].centre = collection[1].centre

    output_path = output_directory / "test.html"
    VisualiseGraph(collection).save(str(output_path))

    assert output_path.exists()


@pytest.fixture(name="multi_wavelength_model")
def make_multi_wavelength_model():
    wavelength_list = [464, 658, 806]

    lens = af.Collection(
        redshift=0.5,
        bulge=af.Gaussian,
        mass=af.Gaussian,
        shear=af.Gaussian,
    )

    source = af.Collection(
        redshift=1.0,
        bulge=af.Exponential,
    )

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    lens_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    lens_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    source_m = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    source_c = af.UniformPrior(lower_limit=-10.0, upper_limit=10.0)

    collection = af.Collection()

    for wavelength in wavelength_list:
        lens_normalization = (wavelength * lens_m) + lens_c
        source_normalization = (wavelength * source_m) + source_c

        collection.append(
            model.replacing(
                {
                    model.galaxies.lens.bulge.normalization: lens_normalization,
                    model.galaxies.source.bulge.normalization: source_normalization,
                }
            )
        )
    return collection


def test_wavelength_info(multi_wavelength_model):
    print(multi_wavelength_model.info)
    assert (
        multi_wavelength_model.info
        == """Total Free Parameters = 14

model                                                                           Collection (N=14)
    0                                                                           Collection (N=14)
        galaxies                                                                Collection (N=14)
            lens                                                                Collection (N=10)
                bulge                                                           Gaussian (N=4)
                    normalization                                               SumPrior (N=2)
                        self                                                    MultiplePrior (N=1)
                mass - shear                                                    Gaussian (N=3)
            source                                                              Collection (N=4)
                bulge                                                           Exponential (N=4)
                    normalization                                               SumPrior (N=2)
                        self                                                    MultiplePrior (N=1)
    1                                                                           Collection (N=14)
        galaxies                                                                Collection (N=14)
            lens                                                                Collection (N=10)
                bulge                                                           Gaussian (N=4)
                    normalization                                               SumPrior (N=2)
                        self                                                    MultiplePrior (N=1)
                mass - shear                                                    Gaussian (N=3)
            source                                                              Collection (N=4)
                bulge                                                           Exponential (N=4)
                    normalization                                               SumPrior (N=2)
                        self                                                    MultiplePrior (N=1)
    2                                                                           Collection (N=14)
        galaxies                                                                Collection (N=14)
            lens                                                                Collection (N=10)
                bulge                                                           Gaussian (N=4)
                    normalization                                               SumPrior (N=2)
                        self                                                    MultiplePrior (N=1)
                mass - shear                                                    Gaussian (N=3)
            source                                                              Collection (N=4)
                bulge                                                           Exponential (N=4)
                    normalization                                               SumPrior (N=2)
                        self                                                    MultiplePrior (N=1)

0 - 2
    galaxies
        lens - source
            bulge - shear
                centre - normalization                                          UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
                sigma                                                           UniformPrior [2], lower_limit = 0.0, upper_limit = 1.0"""
    )


def test_wavelength_visualise(multi_wavelength_model):
    VisualiseGraph(multi_wavelength_model).save("test.html")
