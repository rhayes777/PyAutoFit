from os import listdir
from pathlib import Path

import autofit as af
from autoconf.conf import output_path_for_test
from autofit import Gaussian
from autofit import conf
from autofit.database import Fit
from autofit.tools.update_identifiers import update_directory_identifiers, update_database_identifiers

output_directory = Path(
    __file__
).parent / "output"


@output_path_for_test(
    output_directory
)
def test_directory():
    conf.instance["general"]["output"]["identifier_version"] = 1
    search = af.DynestyStatic(
        name="name"
    )
    search.paths.model = af.PriorModel(
        Gaussian
    )
    search.paths.save_all(
        search_config_dict=search.config_dict_search,
        info={},
        pickle_files=[]
    )

    assert listdir(
        output_directory / "name"
    ) == [
               search.paths.identifier
           ]

    conf.instance["general"]["output"]["identifier_version"] = 3
    update_directory_identifiers(
        output_directory
    )

    print(listdir(
        output_directory / "name"
    ))

    filename, = listdir(
        output_directory / "name"
    )

    identifier, suffix = filename.split(".")
    assert identifier != search.paths.identifier
    assert suffix == "zip"


def test_database(session):
    conf.instance["general"]["output"]["identifier_version"] = 1
    search = af.DynestyStatic(
        name="name",
        session=session
    )
    search.paths.model = af.PriorModel(
        Gaussian
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
