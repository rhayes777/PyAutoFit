from os import listdir
from pathlib import Path

import autofit as af
from autoconf.conf import output_path_for_test
from autofit import Gaussian
from autofit import conf
from autofit.tools.update_identifiers import update_identifiers

output_directory = Path(
    __file__
).parent / "output"


@output_path_for_test(
    output_directory
)
def test_consistent_identifier():
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
    update_identifiers(
        output_directory
    )

    identifier, = listdir(
        output_directory / "name"
    )
    assert identifier != search.paths.identifier
    assert af.SearchOutput(
        output_directory / "name" / identifier
    ).search.paths.identifier == identifier
