import pytest
import numpy as np

import autofit as af
from autoconf.conf import with_config
from autofit import DirectoryPaths, Samples
from autofit.exc import SamplesException
from autofit.non_linear.analysis.latent_variables import LatentVariables


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return 1.0

    def compute_latent_variable(self, instance):
        return {"fwhm": instance.fwhm}


def test_latent_variables():
    latent_variables = LatentVariables()
    latent_variables.add(centre=1.0)

    assert latent_variables.names == ["centre"]
    assert latent_variables.values == [[1.0]]


def test_multiple_quantities():
    latent_variables = LatentVariables()
    latent_variables.add(centre=1.0, intensity=2.0)

    assert latent_variables.names == ["centre", "intensity"]
    assert latent_variables.values == [[1.0, 2.0]]


def test_multiple_iterations():
    latent_variables = LatentVariables()
    latent_variables.add(centre=1.0, intensity=2.0)
    latent_variables.add(centre=3.0, intensity=4.0)

    assert latent_variables.names == ["centre", "intensity"]
    assert latent_variables.values == [[1.0, 2.0], [3.0, 4.0]]


def test_split_addition():
    latent_variables = LatentVariables()
    latent_variables.add(centre=1.0)
    with pytest.raises(SamplesException):
        latent_variables.add(intensity=2.0)


@pytest.fixture(name="latent_variables")
def make_latent_variables():
    return LatentVariables(names=["centre"], values=[[1.0]])


@with_config(
    "general",
    "output",
    "samples_to_csv",
    value=True,
)
def test_set_directory_paths(output_directory, latent_samples):
    directory_paths = DirectoryPaths()
    directory_paths.save_latent_samples(
        latent_samples=latent_samples,
    )
    loaded = directory_paths.load_latent_samples()
    assert len(loaded) == 1


def test_efficient(latent_variables):
    assert latent_variables.efficient().values == np.array([[1.0]])


class MockSamples:
    @property
    def max_log_likelihood_index(self):
        return 0


def test_set_database_paths(session, latent_samples):
    database_paths = af.DatabasePaths(session)
    database_paths.save_latent_samples(
        latent_samples=latent_samples,
    )
    loaded = database_paths.load_latent_samples()
    assert loaded.max_log_likelihood_sample.kwargs == {"fwhm": 7.0644601350928475}


def test_iter(latent_variables):
    assert list(latent_variables) == [{"centre": 1.0}]
    assert latent_variables[0] == {"centre": 1.0}
    assert latent_variables["centre"] == [1.0]


def test_minimise(latent_variables):
    latent_variables.minimise(0)
    assert latent_variables.values == [[1.0]]


@pytest.fixture(name="latent_samples")
def make_latent_samples():
    analysis = Analysis()
    return analysis.compute_latent_samples(
        Samples(
            model=af.Model(af.Gaussian),
            sample_list=[
                af.Sample(
                    log_likelihood=1.0,
                    log_prior=0.0,
                    weight=1.0,
                    kwargs={
                        "centre": 1.0,
                        "normalization": 2.0,
                        "sigma": 3.0,
                    },
                )
            ],
        ),
    )


def test_compute_latent_samples(latent_samples):
    assert latent_samples.sample_list[0].kwargs == {"fwhm": 7.0644601350928475}
    assert latent_samples.model.instance_from_vector([1.0]).fwhm == 1.0
