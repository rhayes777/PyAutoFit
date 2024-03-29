import pytest
import numpy as np

import autofit as af
from autofit import DirectoryPaths
from autofit.exc import SamplesException
from autofit.non_linear.analysis.latent_variables import LatentVariables


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        self.save_latent_variables(centre=instance.centre)
        return 1.0


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


def test_analysis_latent_variables():
    analysis = Analysis()
    instance = af.Gaussian()
    analysis.log_likelihood_function(instance=instance)

    latent_variables = analysis.latent_variables
    assert latent_variables.names == ["centre"]
    assert latent_variables.values == [[instance.centre]]


@pytest.fixture(name="latent_variables")
def make_latent_variables():
    return LatentVariables(names=["centre"], values=[[1.0]])


def test_set_directory_paths(output_directory, latent_variables):
    directory_paths = DirectoryPaths()
    directory_paths.save_latent_variables(
        latent_variables=latent_variables,
        samples=None,
    )
    loaded = directory_paths.load_latent_variables()
    assert loaded.names == ["centre"]
    assert loaded.values == [[1.0]]


def test_efficient(latent_variables):
    assert latent_variables.efficient().values == np.array([[1.0]])


class MockSamples:
    @property
    def max_log_likelihood_index(self):
        return 0


def test_set_database_paths(session, latent_variables):
    database_paths = af.DatabasePaths(session)
    database_paths.save_latent_variables(
        latent_variables=latent_variables, samples=MockSamples()
    )
    loaded = database_paths.load_latent_variables()
    assert loaded.names == ["centre"]
    assert loaded.values == [[1.0]]


def test_iter(latent_variables):
    assert list(latent_variables) == [{"centre": 1.0}]
    assert latent_variables[0] == {"centre": 1.0}
    assert latent_variables["centre"] == [1.0]


def test_minimise(latent_variables):
    latent_variables.minimise(0)
    assert latent_variables.values == [[1.0]]
