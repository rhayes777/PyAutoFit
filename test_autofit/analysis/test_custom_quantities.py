import pytest

import autofit as af
from autofit import DirectoryPaths
from autofit.exc import SamplesException
from autofit.non_linear.analysis.custom_quantities import CustomQuantities


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        self.save_custom_quantities(centre=instance.centre)
        return 1.0


def test_custom_quantities():
    custom_quantities = CustomQuantities()
    custom_quantities.add(centre=1.0)

    assert custom_quantities.names == ["centre"]
    assert custom_quantities.values == [[1.0]]


def test_multiple_quantities():
    custom_quantities = CustomQuantities()
    custom_quantities.add(centre=1.0, intensity=2.0)

    assert custom_quantities.names == ["centre", "intensity"]
    assert custom_quantities.values == [[1.0, 2.0]]


def test_multiple_iterations():
    custom_quantities = CustomQuantities()
    custom_quantities.add(centre=1.0, intensity=2.0)
    custom_quantities.add(centre=3.0, intensity=4.0)

    assert custom_quantities.names == ["centre", "intensity"]
    assert custom_quantities.values == [[1.0, 2.0], [3.0, 4.0]]


def test_split_addition():
    custom_quantities = CustomQuantities()
    custom_quantities.add(centre=1.0)
    with pytest.raises(SamplesException):
        custom_quantities.add(intensity=2.0)


def test_analysis_custom_quantities():
    analysis = Analysis()
    instance = af.Gaussian()
    analysis.log_likelihood_function(instance=instance)

    custom_quantities = analysis.custom_quantities
    assert custom_quantities.names == ["centre"]
    assert custom_quantities.values == [[instance.centre]]


def test_set_directory_paths(output_directory):
    directory_paths = DirectoryPaths(output_path=output_directory)
    custom_quantities = CustomQuantities(names=["centre"], values=[[1.0]])
    directory_paths.save_custom_quantities(
        custom_quantities=custom_quantities,
        samples=None,
    )
    loaded = directory_paths.load_custom_quantities()
    assert loaded.names == ["centre"]
    assert loaded.values == [[1.0]]


def test_set_database_paths(session):
    database_paths = af.DatabasePaths(session)
    custom_quantities = CustomQuantities(names=["centre"], values=[[1.0]])
    database_paths.save_custom_quantities(
        custom_quantities=custom_quantities,
        samples=None,
    )
    loaded = database_paths.load_custom_quantities()
    assert loaded.names == ["centre"]
    assert loaded.values == [[1.0]]
