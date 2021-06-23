from pathlib import Path

import autofit as af
from autoconf.conf import output_path_for_test
from autofit import Gaussian
from autofit import conf

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

    conf.instance["general"]["output"]["identifier_version"] = 3
    search_output = af.SearchOutput(
        search.paths.output_path
    )
    search_output.search.paths._identifier = None
    assert search_output.model is not None
    assert search_output.search.paths.identifier != search.paths.identifier
    print(search.paths.output_path)
