import pytest

from autoconf.class_path import get_class_path
from autofit.aggregator import Aggregator
from pathlib import Path
import autofit as af
from autofit.database.aggregator.info import Info


@pytest.fixture(name="directory")
def make_directory():
    return Path(__file__).parent


def test_without(directory):
    aggregator = Aggregator.from_directory(directory)
    model = list(aggregator)[0].model
    assert model.cls is af.Gaussian


def test_with():
    aggregator = Aggregator.from_directory(
        Path(__file__).parent,
        reference={"": get_class_path(af.Exponential)},
    )
    output = list(aggregator)[0]
    model = output.model
    assert model.cls is af.Exponential


@pytest.fixture(name="database_aggregator")
def database_aggregator(
    directory,
    session,
    output_directory,
):
    aggregator = af.Aggregator(
        session,
        filename=output_directory / "database.sqlite",
    )
    aggregator.add_directory(
        directory,
        reference={"": get_class_path(af.Exponential)},
    )

    return aggregator


def test_database(database_aggregator):
    fit = list(database_aggregator)[0]
    model = fit.model
    assert model.cls is af.Exponential


@pytest.fixture(name="info")
def make_info(database_aggregator):
    return Info(database_aggregator.session)


def test_query_fits(info):
    fits = info.fits
    assert len(info.fits) == 3
    assert fits[0].total_parameters == 4


def test_headers_and_rows(info):
    assert len(info.headers) == len(info.rows[0])


def test_database_info(
    database_aggregator,
    output_directory,
):
    print((output_directory / "database.info").read_text())
    assert (
        (output_directory / "database.info").read_text()
        == """                         unique_id,name,unique_tag,total_free_parameters,is_complete
  c4bf344d706947aa66b129ed2e05e1bd,    ,          ,                    4,      False
c4bf344d706947aa66b129ed2e05e1bd_0,    ,          ,                    0,           
c4bf344d706947aa66b129ed2e05e1bd_1,    ,          ,                    0,           
"""
    )
