import zipfile
from os import listdir
from pathlib import Path

import pytest

import autofit as af
from autoconf.conf import output_path_for_test
from autofit import Gaussian
from autofit import conf
from autofit.database import Fit
from autofit.tools.update_identifiers import (
    update_directory_identifiers,
    update_database_identifiers,
    update_identifiers_from_dict
)

output_directory = Path(
    __file__
).parent / "output"


@output_path_for_test(
    output_directory,
)
def test_directory(
        old_directory_paths
):
    old_directory_paths.save_all()

    assert listdir(
        output_directory / "name"
    ) == [
               old_directory_paths.identifier
           ]

    conf.instance["general"]["output"]["identifier_version"] = 3
    update_directory_identifiers(
        output_directory
    )

    filename, = listdir(
        output_directory / "name"
    )

    identifier, suffix = filename.split(".")
    assert identifier != old_directory_paths.identifier
    assert suffix == "zip"


@pytest.fixture(
    name="old_directory_paths"
)
def make_old_directory_paths():
    conf.instance["general"]["output"]["identifier_version"] = 1
    search = af.DynestyStatic(
        name="name"
    )
    search.paths.model = af.PriorModel(
        af.Gaussian
    )
    return search.paths


@output_path_for_test(
    output_directory
)
def test_update_identifiers_from_dict_no_change():
    search = af.DynestyStatic(
        name="name"
    )
    search.paths.model = af.PriorModel(
        af.Gaussian
    )
    old_directory_paths = search.paths

    old_directory_paths.save_all()
    old_directory_paths.zip_remove()

    update_identifiers_from_dict(
        output_directory,
        {}
    )

    filename, = listdir(
        output_directory / "name"
    )

    identifier, suffix = filename.split(".")
    assert identifier == old_directory_paths.identifier
    assert suffix == "zip"


@output_path_for_test(
    output_directory
)
def test_update_identifiers_from_dict():
    search = af.DynestyStatic(
        name="name"
    )
    search.paths.model = af.PriorModel(
        af.Gaussian
    )
    old_directory_paths = search.paths

    initial_length = len(
        old_directory_paths._identifier.hash_list
    )

    old_directory_paths.save_all()
    old_directory_paths.zip_remove()

    update_identifiers_from_dict(
        output_directory,
        {
            "normalization": "magnitude"
        }
    )

    filename, = listdir(
        output_directory / "name"
    )

    identifier, suffix = filename.split(".")
    assert identifier != old_directory_paths.identifier
    assert suffix == "zip"

    unzipped = output_directory / "unzipped"
    with zipfile.ZipFile(output_directory / "name" / filename, "r") as f:
        f.extractall(unzipped)

    with open(
            unzipped / ".identifier"
    ) as f:
        lines = f.read().split("\n")
        assert "normalization" not in lines
        assert "magnitude" in lines

    assert len(lines) == initial_length


@output_path_for_test(
    output_directory,
)
def test_zipped_no_change(
        old_directory_paths
):
    old_directory_paths.save_all()
    old_directory_paths.zip_remove()

    update_directory_identifiers(
        output_directory
    )

    filename, = listdir(
        output_directory / "name"
    )

    identifier, suffix = filename.split(".")
    assert identifier == old_directory_paths.identifier
    assert suffix == "zip"


@output_path_for_test(
    output_directory,
)
def test_zipped(
        old_directory_paths
):
    old_directory_paths.save_all()
    old_directory_paths.zip_remove()

    assert listdir(
        output_directory / "name"
    ) == [
               f"{old_directory_paths.identifier}.zip"
           ]

    conf.instance["general"]["output"]["identifier_version"] = 3
    update_directory_identifiers(
        output_directory
    )

    filename, = listdir(
        output_directory / "name"
    )

    identifier, suffix = filename.split(".")
    assert identifier != old_directory_paths.identifier
    assert suffix == "zip"


def test_database(session):
    conf.instance["general"]["output"]["identifier_version"] = 1
    search = af.DynestyStatic(
        name="name",
        session=session
    )
    search.paths.model = af.PriorModel(
        af.Gaussian
    )
    search.paths.save_all(
        search_config_dict=search.config_dict_search,
        info={},
        pickle_files=[]
    )

    fit, = Fit.all(
        session=session
    )
    assert fit.id == search.paths.identifier

    conf.instance["general"]["output"]["identifier_version"] = 3
    update_database_identifiers(
        session
    )

    fit, = Fit.all(
        session=session
    )
    assert fit.id != search.paths.identifier
