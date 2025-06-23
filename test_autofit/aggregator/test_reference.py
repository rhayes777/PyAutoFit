import os

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
    model_list = [agg.model for agg in aggregator]

    assert any([getattr(model, "cls", False) is af.Gaussian for model in model_list])


def test_with():

    aggregator = Aggregator.from_directory(
        Path(__file__).parent,
        reference={"": get_class_path(af.Exponential)},
    )
    output_list = list(aggregator)
    model_list = [output.model for output in output_list]
    assert any([getattr(model, "cls", False) is af.Exponential for model in model_list])

@pytest.fixture(name="database_path")
def database_path(output_directory):
    database_path = output_directory / "database.sqlite"
    yield database_path
    os.remove(database_path)


@pytest.fixture(name="database_aggregator")
def database_aggregator(
    directory,
    database_path,
):
    aggregator = af.Aggregator.from_database(
        database_path,
    )
    aggregator.add_directory(
        directory,
        reference={"": get_class_path(af.Exponential)},
        completed_only=True,
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


def test_info_path(info, output_directory):
    assert info.path == output_directory / "database.info"


def test_database_info(
    database_aggregator,
    output_directory,
):
    print((output_directory / "database.info").read_text())
    assert (
        (output_directory / "database.info").read_text()
        == """                         unique_id,name,unique_tag,total_free_parameters,is_complete
  d05be1e6380082adea5c918af392d2b9,    ,          ,                    4,       True
d05be1e6380082adea5c918af392d2b9_0,    ,          ,                    0,           
d05be1e6380082adea5c918af392d2b9_1,    ,          ,                    0,           
"""
    )
